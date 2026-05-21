/*
 * Triton dynamic_func C accelerator
 * =================================
 *
 * Background
 * ----------
 * Each launch of a Triton kernel runs through an exec'd Python helper called
 * ``dynamic_func`` whose body looks like::
 *
 *     def dynamic_func(arg0, arg1, ..., **options):
 *         params = {'arg0': arg0, 'arg1': arg1, ...}
 *         specialization = [
 *             specialize_impl(arg0, is_const=..., specialize_value=..., align=...),
 *             specialize_impl(arg1, is_const=..., specialize_value=..., align=...),
 *             ...
 *         ]
 *         return params, specialization, options
 *
 * In aarch64 / Grace this is ~22 µs of pure-Python work per launch plus ~8 µs
 * of bytecode gap.  Every transition into Python is also a place where the GIL
 * can be stolen by another in-process thread, so cutting Python overhead also
 * shrinks the GIL-vulnerable window.
 *
 * This module exposes a single PyObject ``BinderState`` whose call protocol is
 * bit-identical to the exec'd ``dynamic_func``.  Construction takes the per-
 * kernel signature (parameter names, flags, annotations, defaults) once; the
 * call protocol then runs entirely in C.  Per-arg specialisation is still
 * delegated to a Python callback (``specialize_impl``) so that uncommon types
 * (TensorDescriptor, GluonTensorDescriptor, tuples, custom dtypes) keep their
 * existing semantics; the C side only fast-paths the boring scalar cases.
 *
 * The fallback is fully transparent: if anything in the C path raises or
 * cannot handle a particular argument, we propagate the exception.  The Python
 * caller (`JITFunction.run`) checks whether the C binder is available before
 * using it; if not it falls back to the existing exec'd Python function.
 *
 * NOTE: standard CPython C API only.  No C++.  No third-party deps.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* ------------------------------------------------------------------------- */
/*   Module-level interned strings and pre-built tuples                       */
/* ------------------------------------------------------------------------- */

static PyObject *S_constexpr;     /* "constexpr"    */
static PyObject *S_u1;            /* "u1"           */
static PyObject *S_i32;           /* "i32"          */
static PyObject *S_i64;           /* "i64"          */
static PyObject *S_u64;           /* "u64"          */
static PyObject *S_fp32;          /* "fp32"         */
static PyObject *S_nvTmaDesc;     /* "nvTmaDesc"    */
static PyObject *S_int;           /* "int"          */
static PyObject *S_tensor;        /* "tensor"       */
static PyObject *S_align;         /* "align"        */
static PyObject *S_data_ptr;      /* "data_ptr"     */
static PyObject *S_dtype;         /* "dtype"        */
static PyObject *S_tma_desc;      /* "tma_desc_cpu_ptr" */
static PyObject *S_star_k;        /* "*k"           */
static PyObject *S_star;          /* "*"            */
static PyObject *S_D;             /* "D"            */
static PyObject *S_empty;         /* ""             */
static PyObject *S_const_str;     /* "const"        */
static PyObject *S_star_k_pref;   /* "*k"           */

static PyObject *T_constexpr_None;  /* ("constexpr", None) */
static PyObject *T_constexpr_1;     /* ("constexpr", 1)    */
static PyObject *T_u1_None;         /* ("u1", None)        */
static PyObject *T_fp32_None;       /* ("fp32", None)      */
static PyObject *T_i32_None;        /* ("i32", None)       */
static PyObject *T_i64_None;        /* ("i64", None)       */
static PyObject *T_u64_None;        /* ("u64", None)       */
static PyObject *T_nvTmaDesc_None;  /* ("nvTmaDesc", None) */

#define I32_LO_LL  ((long long)(-(1LL << 31)))
#define I32_HI_LL  ((long long)((1LL << 31) - 1))

/* Param-flag bits. Kept in sync with the Python side. */
#define PF_IS_CONSTEXPR                 (1u << 0)
#define PF_IS_CONST                     (1u << 1)
#define PF_DO_NOT_SPECIALIZE            (1u << 2)
#define PF_DO_NOT_SPECIALIZE_ON_ALIGN   (1u << 3)
#define PF_HAS_ANNOTATION               (1u << 4)
#define PF_ANNOTATION_NO_SPEC           (1u << 5)  /* annotation is bool/float */
#define PF_HAS_DEFAULT                  (1u << 6)

/* Fast-path bits.
 * SPEC_EXTRA_INLINE: ``backend.get_arg_specialization`` is the unmodified
 * BaseBackend implementation (NVIDIA path).  Replicate it in C without
 * crossing back into Python.  AMD with buffer-ops, or any user-supplied
 * backend that subclasses it, leaves this bit clear and forces the C side
 * to call the Python callback per-arg. */
#define FP_SPEC_EXTRA_INLINE (1u << 2)

/* ------------------------------------------------------------------------- */
/*   BinderState type                                                         */
/* ------------------------------------------------------------------------- */

typedef struct {
    PyObject *name;            /* interned str (borrowed from the tuple) */
    PyObject *default_value;   /* NULL if no default; else strong ref    */
    PyObject *annotation;      /* NULL or strong ref to annotation str   */
    unsigned int flags;
} ParamSpec;

typedef struct {
    PyObject_HEAD
    Py_ssize_t n_params;
    ParamSpec *params;
    PyObject *param_names_tuple;     /* tuple of all names, strong ref          */
    PyObject *specialize_impl;       /* Python fallback callable                */
    PyObject *specialize_extra;      /* backend.get_arg_specialization callable */
    PyObject *dtype2str;             /* shared dict                              */
    unsigned int fast_flags;
} BinderState;

/* -- helpers ------------------------------------------------------------- */

static int
binder_check_args(BinderState *self) {
    if (!self->specialize_impl || !PyCallable_Check(self->specialize_impl)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "BinderState: specialize_impl callable missing");
        return -1;
    }
    if (!self->specialize_extra || !PyCallable_Check(self->specialize_extra)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "BinderState: specialize_extra callable missing");
        return -1;
    }
    return 0;
}

/* Build a fresh 2-tuple ``(type_obj, key_obj)``, taking strong refs.
 * Returns a new reference. */
static PyObject *
make_pair(PyObject *type_obj, PyObject *key_obj) {
    PyObject *t = PyTuple_New(2);
    if (!t) return NULL;
    Py_INCREF(type_obj);
    Py_INCREF(key_obj);
    PyTuple_SET_ITEM(t, 0, type_obj);
    PyTuple_SET_ITEM(t, 1, key_obj);
    return t;
}

/* Inline reproduction of ``BaseBackend.get_arg_specialization`` for the
 * NVIDIA / default backend (and any subclass that didn't override the
 * default).  The Python source (3rdparty/triton/python/triton/backends/
 * compiler.py) is::
 *
 *     if ty == "int" and arg % 16 == 0 and kwargs.get("align", False):
 *         return "D"
 *     if ty == "tensor" and arg.data_ptr() % 16 == 0 and kwargs.get("align", False):
 *         return "D"
 *     return ""
 *
 * Returns a *new* reference to ``S_D`` or ``S_empty``, or NULL on error.
 * ``ty`` must be S_int or S_tensor.
 */
static PyObject *
inline_spec_extra(PyObject *arg, PyObject *ty, int align) {
    if (!align) {
        Py_INCREF(S_empty);
        return S_empty;
    }
    if (ty == S_int) {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(arg, &overflow);
        if (v == -1 && PyErr_Occurred()) {
            return NULL;
        }
        if (overflow == 0) {
            /* v in [INT64_MIN, INT64_MAX] -- simple modulo. */
            if ((v & 15LL) == 0) { Py_INCREF(S_D); return S_D; }
            Py_INCREF(S_empty); return S_empty;
        }
        /* Big int: do mod via PyNumber.  Rare path (offsets > 2**63). */
        PyObject *sixteen = PyLong_FromLong(16);
        if (!sixteen) return NULL;
        PyObject *mod = PyNumber_Remainder(arg, sixteen);
        Py_DECREF(sixteen);
        if (!mod) return NULL;
        int is_zero = PyObject_Not(mod);
        Py_DECREF(mod);
        if (is_zero < 0) return NULL;
        if (is_zero) { Py_INCREF(S_D); return S_D; }
        Py_INCREF(S_empty); return S_empty;
    }
    if (ty == S_tensor) {
        PyObject *ptr_obj = PyObject_CallMethodObjArgs(arg, S_data_ptr, NULL);
        if (!ptr_obj) return NULL;
        unsigned long long uptr;
        if (PyLong_Check(ptr_obj)) {
            uptr = PyLong_AsUnsignedLongLongMask(ptr_obj);
            Py_DECREF(ptr_obj);
            if (PyErr_Occurred()) return NULL;
        } else {
            Py_DECREF(ptr_obj);
            Py_INCREF(S_empty);
            return S_empty;
        }
        if ((uptr & 15ULL) == 0) { Py_INCREF(S_D); return S_D; }
        Py_INCREF(S_empty); return S_empty;
    }
    Py_INCREF(S_empty);
    return S_empty;
}

/* Call ``specialize_extra(arg, ty, align=align)`` -- either inlined (base
 * backend) or by dispatching to the Python callback.  ``ty`` is interned
 * S_int / S_tensor.  Returns new reference. */
static PyObject *
call_specialize_extra(BinderState *self, PyObject *arg, PyObject *ty, int align) {
    if (self->fast_flags & FP_SPEC_EXTRA_INLINE) {
        return inline_spec_extra(arg, ty, align);
    }
    /* Generic path -- vectorcall preferred for speed, fall back to
     * PyObject_Call when vectorcall is unavailable. */
    PyObject *align_obj = align ? Py_True : Py_False;
    PyObject *stack[3] = {arg, ty, align_obj};
    PyObject *kw_names = PyTuple_New(1);
    if (!kw_names) return NULL;
    Py_INCREF(S_align);
    PyTuple_SET_ITEM(kw_names, 0, S_align);
    PyObject *r = PyObject_Vectorcall(self->specialize_extra, stack, 2, kw_names);
    Py_DECREF(kw_names);
    return r;
}

/* Fall back to the legacy Python ``specialize_impl(arg, is_const, specialize_value, align)``.
 * Returns a new reference (a 2-tuple). */
static PyObject *
call_specialize_impl(BinderState *self, PyObject *arg, int is_const,
                     int specialize_value, int align) {
    PyObject *args = PyTuple_New(4);
    if (!args) return NULL;
    Py_INCREF(arg);
    PyTuple_SET_ITEM(args, 0, arg);
    PyTuple_SET_ITEM(args, 1, PyBool_FromLong(is_const));
    PyTuple_SET_ITEM(args, 2, PyBool_FromLong(specialize_value));
    PyTuple_SET_ITEM(args, 3, PyBool_FromLong(align));
    PyObject *r = PyObject_CallObject(self->specialize_impl, args);
    Py_DECREF(args);
    return r;
}

/* Specialise a single arg.  ``flags`` is the per-param flag bitfield (only the
 * is_const / do_not_specialize / do_not_specialize_on_alignment / has_annotation
 * bits are consulted; is_constexpr is handled by the caller).
 *
 * Returns a new reference to a 2-tuple (type, key), or NULL on error.
 *
 * The C fast paths cover:
 *   None, bool, int (full range), float, has-data_ptr tensor,
 *   tma_desc_cpu_ptr.
 *
 * Tuples, TensorDescriptor, GluonTensorDescriptor, JITFunction, constexpr,
 * and anything we don't recognise are punted to the Python specialize_impl
 * callback.  This keeps semantics identical for the rare cases without
 * having to re-implement them here.
 */
static PyObject *
specialise_one(BinderState *self, PyObject *arg, unsigned int flags,
               PyObject *annotation) {
    int is_const = (flags & PF_IS_CONST) ? 1 : 0;
    int specialize_value = (flags & PF_DO_NOT_SPECIALIZE) ? 0 : 1;
    int align = (flags & PF_DO_NOT_SPECIALIZE_ON_ALIGN) ? 0 : 1;

    /* Annotation path overrides the type half. */
    if (flags & PF_HAS_ANNOTATION) {
        /* The legacy logic:
         *   if annotation_type in str and starts with u1/fp/bf: specialize = False
         *   if specialize:  ("ann",) + spec[1:]
         *   else:           ("ann", None)
         */
        if (flags & PF_ANNOTATION_NO_SPEC) {
            /* annotation is "u1" / "fp..." / "bf..." -> drop the spec key. */
            return make_pair(annotation, Py_None);
        }
        /* Need to compute the regular spec just for its key half. */
        PyObject *inner = call_specialize_impl(self, arg, is_const,
                                               specialize_value, align);
        if (!inner) return NULL;
        if (!PyTuple_Check(inner) || PyTuple_GET_SIZE(inner) < 2) {
            Py_DECREF(inner);
            PyErr_SetString(PyExc_RuntimeError,
                            "specialize_impl did not return a 2-tuple");
            return NULL;
        }
        PyObject *key = PyTuple_GET_ITEM(inner, 1);
        PyObject *pair = make_pair(annotation, key);
        Py_DECREF(inner);
        return pair;
    }

    /* No annotation -> the regular spec branch decides everything. */

    /* None -> ("constexpr", None) */
    if (arg == Py_None) {
        Py_INCREF(T_constexpr_None);
        return T_constexpr_None;
    }

    /* bool -- MUST precede the int check (PyBool is a PyLong subclass). */
    if (PyBool_Check(arg)) {
        Py_INCREF(T_u1_None);
        return T_u1_None;
    }

    /* int */
    if (PyLong_Check(arg)) {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(arg, &overflow);
        if (v == -1 && PyErr_Occurred()) {
            return NULL;
        }

        /* General path. */
        PyObject *key;
        if (specialize_value) {
            key = call_specialize_extra(self, arg, S_int, align);
            if (!key) return NULL;
        } else {
            Py_INCREF(Py_None);
            key = Py_None;
        }

        PyObject *result = NULL;
        if (overflow == 0 && v == 1 && specialize_value) {
            Py_DECREF(key);
            Py_INCREF(T_constexpr_1);
            result = T_constexpr_1;
        } else if (overflow == 0 && v >= I32_LO_LL && v <= I32_HI_LL) {
            result = make_pair(S_i32, key);
            Py_DECREF(key);
        } else if (overflow > 0) {
            /* Positive overflow: candidate for the u64 window. */
            unsigned long long uv = PyLong_AsUnsignedLongLongMask(arg);
            if (PyErr_Occurred()) {
                /* Genuinely huge: fall back to i64. */
                PyErr_Clear();
                result = make_pair(S_i64, key);
            } else if (uv >= (1ULL << 63)) {
                result = make_pair(S_u64, key);
            } else {
                result = make_pair(S_i64, key);
            }
            Py_DECREF(key);
        } else {
            /* Negative overflow or in-range but outside i32 -> i64. */
            result = make_pair(S_i64, key);
            Py_DECREF(key);
        }
        return result;
    }

    /* float */
    if (PyFloat_Check(arg)) {
        Py_INCREF(T_fp32_None);
        return T_fp32_None;
    }

    /* Tensor-like: anything that has a ``data_ptr`` attribute. */
    if (PyObject_HasAttr(arg, S_data_ptr)) {
        PyObject *dtype = PyObject_GetAttr(arg, S_dtype);
        if (!dtype) {
            PyErr_Clear();
            /* Fall back to Python -- this is an odd object. */
            return call_specialize_impl(self, arg, is_const, specialize_value, align);
        }

        /* Look up the cached canonicalised type string. */
        PyObject *dtype_key = PyTuple_Pack(2, dtype, is_const ? Py_True : Py_False);
        Py_DECREF(dtype);
        if (!dtype_key) return NULL;
        PyObject *type_str = NULL;
        if (self->dtype2str) {
            type_str = PyDict_GetItem(self->dtype2str, dtype_key);  /* borrowed */
        }
        if (!type_str) {
            /* The Python side maintains the cache and populates it lazily.
             * Punt so we don't have to re-implement canonicalize_dtype here. */
            Py_DECREF(dtype_key);
            return call_specialize_impl(self, arg, is_const, specialize_value, align);
        }
        Py_INCREF(type_str);
        Py_DECREF(dtype_key);

        /* Compute the key half. */
        PyObject *key;
        if (specialize_value) {
            key = call_specialize_extra(self, arg, S_tensor, align);
            if (!key) {
                Py_DECREF(type_str);
                return NULL;
            }
        } else {
            Py_INCREF(Py_None);
            key = Py_None;
        }
        PyObject *pair = make_pair(type_str, key);
        Py_DECREF(type_str);
        Py_DECREF(key);
        return pair;
    }

    /* tma_desc_cpu_ptr  -> ("nvTmaDesc", None) */
    if (PyObject_HasAttr(arg, S_tma_desc)) {
        Py_INCREF(T_nvTmaDesc_None);
        return T_nvTmaDesc_None;
    }

    /* Tuples, TensorDescriptor, GluonTensorDescriptor, JITFunction, constexpr,
     * anything else -> hand back to Python. */
    return call_specialize_impl(self, arg, is_const, specialize_value, align);
}

/* ------------------------------------------------------------------------- */
/*   BinderState.__call__                                                     */
/* ------------------------------------------------------------------------- */

static PyObject *
BinderState_call(BinderState *self, PyObject *args, PyObject *kwargs) {
    if (binder_check_args(self) < 0) return NULL;

    Py_ssize_t n_args = PyTuple_GET_SIZE(args);
    Py_ssize_t n_kwargs = kwargs ? PyDict_GET_SIZE(kwargs) : 0;
    Py_ssize_t n_params = self->n_params;

    /* Build params dict, specialisation list, options dict. */
    PyObject *params = PyDict_New();
    PyObject *specialization = PyList_New(n_params);
    if (!params || !specialization) {
        Py_XDECREF(params); Py_XDECREF(specialization);
        return NULL;
    }

    /* options is a *copy* of kwargs that we will prune as we consume names.
     * Use a fresh dict so the caller's kwargs object stays unmodified. */
    PyObject *options;
    if (kwargs && n_kwargs > 0) {
        options = PyDict_Copy(kwargs);
        if (!options) {
            Py_DECREF(params); Py_DECREF(specialization);
            return NULL;
        }
    } else {
        options = PyDict_New();
        if (!options) {
            Py_DECREF(params); Py_DECREF(specialization);
            return NULL;
        }
    }

    for (Py_ssize_t i = 0; i < n_params; i++) {
        ParamSpec *p = &self->params[i];

        /* 1. Resolve the actual arg value for this parameter. */
        PyObject *value = NULL;
        int from_options = 0;

        if (i < n_args) {
            value = PyTuple_GET_ITEM(args, i);  /* borrowed */
            Py_INCREF(value);
        } else {
            /* Try options[name].  PyDict_GetItemWithError returns borrowed. */
            PyObject *kw_value = NULL;
            if (PyDict_GET_SIZE(options) > 0) {
                kw_value = PyDict_GetItemWithError(options, p->name);
                if (!kw_value && PyErr_Occurred()) {
                    goto error;
                }
            }
            if (kw_value) {
                Py_INCREF(kw_value);
                value = kw_value;
                from_options = 1;
            } else if (p->flags & PF_HAS_DEFAULT) {
                Py_INCREF(p->default_value);
                value = p->default_value;
            } else {
                PyErr_Format(PyExc_TypeError,
                             "dynamic_func() missing required argument: '%U'",
                             p->name);
                goto error;
            }
        }

        /* 2. Pop from options if we consumed a kwarg. */
        if (from_options) {
            if (PyDict_DelItem(options, p->name) < 0) {
                Py_DECREF(value);
                goto error;
            }
        }

        /* 3. Store in params dict. */
        if (PyDict_SetItem(params, p->name, value) < 0) {
            Py_DECREF(value);
            goto error;
        }

        /* 4. Compute specialisation for this arg. */
        PyObject *spec_entry;
        if (p->flags & PF_IS_CONSTEXPR) {
            spec_entry = make_pair(S_constexpr, value);
        } else {
            spec_entry = specialise_one(self, value, p->flags, p->annotation);
        }
        Py_DECREF(value);
        if (!spec_entry) {
            goto error;
        }
        /* PyList_SET_ITEM steals the reference. */
        PyList_SET_ITEM(specialization, i, spec_entry);
    }

    /* Pack into the 3-tuple expected by ``JITFunction.run``. */
    PyObject *result = PyTuple_New(3);
    if (!result) {
        goto error;
    }
    PyTuple_SET_ITEM(result, 0, params);
    PyTuple_SET_ITEM(result, 1, specialization);
    PyTuple_SET_ITEM(result, 2, options);
    return result;

error:
    Py_DECREF(params);
    Py_DECREF(specialization);
    Py_DECREF(options);
    return NULL;
}

/* ------------------------------------------------------------------------- */
/*   BinderState.tp_dealloc / tp_init                                         */
/* ------------------------------------------------------------------------- */

static void
BinderState_dealloc(BinderState *self) {
    if (self->params) {
        for (Py_ssize_t i = 0; i < self->n_params; i++) {
            Py_XDECREF(self->params[i].default_value);
            Py_XDECREF(self->params[i].annotation);
        }
        PyMem_Free(self->params);
    }
    Py_XDECREF(self->param_names_tuple);
    Py_XDECREF(self->specialize_impl);
    Py_XDECREF(self->specialize_extra);
    Py_XDECREF(self->dtype2str);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
BinderState_init(BinderState *self, PyObject *args, PyObject *kwargs) {
    /* Construction protocol:
     *   BinderState(param_names, param_flags, annotations, defaults,
     *               specialize_impl, specialize_extra, dtype2str, fast_flags)
     *
     *   param_names  : tuple[str]
     *   param_flags  : tuple[int]   (bitfield per param)
     *   annotations  : tuple[Optional[str]]
     *   defaults     : tuple[Optional[obj]]   (None ok, see PF_HAS_DEFAULT)
     *   specialize_impl  : callable
     *   specialize_extra : callable
     *   dtype2str    : dict
     *   fast_flags   : int
     */
    static char *kwlist[] = {
        "param_names", "param_flags", "annotations", "defaults",
        "specialize_impl", "specialize_extra", "dtype2str", "fast_flags",
        NULL
    };
    PyObject *names = NULL, *flags = NULL, *anns = NULL, *defs = NULL;
    PyObject *sp_impl = NULL, *sp_extra = NULL, *d2s = NULL;
    unsigned int fast_flags = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOOOOOI:BinderState", kwlist,
            &names, &flags, &anns, &defs,
            &sp_impl, &sp_extra, &d2s, &fast_flags)) {
        return -1;
    }
    if (!PyTuple_Check(names) || !PyTuple_Check(flags) ||
        !PyTuple_Check(anns)  || !PyTuple_Check(defs)) {
        PyErr_SetString(PyExc_TypeError,
                        "BinderState: param_names/flags/annotations/defaults must be tuples");
        return -1;
    }
    if (!PyCallable_Check(sp_impl) || !PyCallable_Check(sp_extra) ||
        !PyDict_Check(d2s)) {
        PyErr_SetString(PyExc_TypeError,
                        "BinderState: specialize_impl/specialize_extra must be callable, dtype2str must be dict");
        return -1;
    }
    Py_ssize_t n = PyTuple_GET_SIZE(names);
    if (PyTuple_GET_SIZE(flags) != n || PyTuple_GET_SIZE(anns) != n ||
        PyTuple_GET_SIZE(defs)  != n) {
        PyErr_SetString(PyExc_ValueError,
                        "BinderState: all parameter tuples must have equal length");
        return -1;
    }

    ParamSpec *ps = NULL;
    if (n > 0) {
        ps = (ParamSpec *)PyMem_Calloc((size_t)n, sizeof(ParamSpec));
        if (!ps) {
            PyErr_NoMemory();
            return -1;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *nm = PyTuple_GET_ITEM(names, i);
        PyObject *fl = PyTuple_GET_ITEM(flags, i);
        PyObject *an = PyTuple_GET_ITEM(anns,  i);
        PyObject *dv = PyTuple_GET_ITEM(defs,  i);
        if (!PyUnicode_Check(nm)) {
            PyErr_SetString(PyExc_TypeError, "BinderState: param_names must contain str");
            goto bad;
        }
        if (!PyLong_Check(fl)) {
            PyErr_SetString(PyExc_TypeError, "BinderState: param_flags must contain int");
            goto bad;
        }
        ps[i].name = nm;          /* borrowed: kept alive by param_names_tuple */
        ps[i].flags = (unsigned int)PyLong_AsUnsignedLongMask(fl);
        if (an != Py_None) {
            if (!PyUnicode_Check(an)) {
                PyErr_SetString(PyExc_TypeError,
                                "BinderState: annotations entries must be str or None");
                goto bad;
            }
            Py_INCREF(an);
            ps[i].annotation = an;
        }
        if (ps[i].flags & PF_HAS_DEFAULT) {
            Py_INCREF(dv);
            ps[i].default_value = dv;
        }
    }

    self->n_params = n;
    self->params = ps;
    Py_INCREF(names);     self->param_names_tuple = names;
    Py_INCREF(sp_impl);   self->specialize_impl   = sp_impl;
    Py_INCREF(sp_extra);  self->specialize_extra  = sp_extra;
    Py_INCREF(d2s);       self->dtype2str         = d2s;
    self->fast_flags = fast_flags;
    return 0;

bad:
    if (ps) {
        for (Py_ssize_t i = 0; i < n; i++) {
            Py_XDECREF(ps[i].default_value);
            Py_XDECREF(ps[i].annotation);
        }
        PyMem_Free(ps);
    }
    return -1;
}

static PyTypeObject BinderState_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_binder_c.BinderState",
    .tp_basicsize = sizeof(BinderState),
    .tp_dealloc   = (destructor)BinderState_dealloc,
    .tp_call      = (ternaryfunc)BinderState_call,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Compiled per-kernel argument binder; call protocol matches "
                    "the legacy dynamic_func.",
    .tp_init      = (initproc)BinderState_init,
    .tp_new       = PyType_GenericNew,
};

/* ------------------------------------------------------------------------- */
/*   Module init                                                              */
/* ------------------------------------------------------------------------- */

static PyMethodDef BinderMethods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef binder_module = {
    PyModuleDef_HEAD_INIT,
    "_binder_c",
    "Triton dynamic_func C accelerator",
    -1,
    BinderMethods,
    NULL, NULL, NULL, NULL
};

#define INTERN_OR_FAIL(var, lit) do { \
    var = PyUnicode_InternFromString(lit); \
    if (!var) return NULL; \
} while (0)

PyMODINIT_FUNC
PyInit__binder_c(void) {
    if (PyType_Ready(&BinderState_Type) < 0) return NULL;

    PyObject *m = PyModule_Create(&binder_module);
    if (!m) return NULL;

    /* Intern all the recurring strings once. */
    INTERN_OR_FAIL(S_constexpr,   "constexpr");
    INTERN_OR_FAIL(S_u1,          "u1");
    INTERN_OR_FAIL(S_i32,         "i32");
    INTERN_OR_FAIL(S_i64,         "i64");
    INTERN_OR_FAIL(S_u64,         "u64");
    INTERN_OR_FAIL(S_fp32,        "fp32");
    INTERN_OR_FAIL(S_nvTmaDesc,   "nvTmaDesc");
    INTERN_OR_FAIL(S_int,         "int");
    INTERN_OR_FAIL(S_tensor,      "tensor");
    INTERN_OR_FAIL(S_align,       "align");
    INTERN_OR_FAIL(S_data_ptr,    "data_ptr");
    INTERN_OR_FAIL(S_dtype,       "dtype");
    INTERN_OR_FAIL(S_tma_desc,    "tma_desc_cpu_ptr");
    INTERN_OR_FAIL(S_star_k,      "*k");
    INTERN_OR_FAIL(S_star,        "*");
    INTERN_OR_FAIL(S_D,           "D");
    INTERN_OR_FAIL(S_empty,       "");
    INTERN_OR_FAIL(S_const_str,   "const");
    INTERN_OR_FAIL(S_star_k_pref, "*k");

    T_constexpr_None  = PyTuple_Pack(2, S_constexpr, Py_None);
    PyObject *one_py = PyLong_FromLong(1);
    if (!one_py) return NULL;
    T_constexpr_1    = PyTuple_Pack(2, S_constexpr, one_py);
    Py_DECREF(one_py);
    T_u1_None        = PyTuple_Pack(2, S_u1,  Py_None);
    T_fp32_None      = PyTuple_Pack(2, S_fp32, Py_None);
    T_i32_None       = PyTuple_Pack(2, S_i32,  Py_None);
    T_i64_None       = PyTuple_Pack(2, S_i64,  Py_None);
    T_u64_None       = PyTuple_Pack(2, S_u64,  Py_None);
    T_nvTmaDesc_None = PyTuple_Pack(2, S_nvTmaDesc, Py_None);
    if (!T_constexpr_None || !T_constexpr_1 || !T_u1_None || !T_fp32_None ||
        !T_i32_None || !T_i64_None || !T_u64_None || !T_nvTmaDesc_None) {
        return NULL;
    }

    Py_INCREF(&BinderState_Type);
    if (PyModule_AddObject(m, "BinderState", (PyObject *)&BinderState_Type) < 0) {
        Py_DECREF(&BinderState_Type);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
