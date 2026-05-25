"""
CUDA Emitter: Translates TTGIR/NVGPUIR MLIR modules to CUDA C++ source code.

This module implements the core translation from Triton's optimized GPU IR
to readable, compilable CUDA code. It works at the TTGIR level where all
GPU-specific optimizations (tiling, coalescing, pipelining, etc.) have
already been applied, but high-level control flow structures (for loops,
if/else) are still preserved.

Architecture:
    MLIRTextParser  -> parses MLIR text to internal IR representation
    LayoutComputer  -> computes per-thread element mapping from layout encodings
    CUDACodeGen     -> generates CUDA C++ code from parsed IR
    CUDAEmitter     -> top-level orchestrator
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Internal IR Representation
# ---------------------------------------------------------------------------

@dataclass
class BlockedLayout:
    size_per_thread: List[int]
    threads_per_warp: List[int]
    warps_per_cta: List[int]
    order: List[int]
    ctas_per_cga: List[int] = field(default_factory=lambda: [1, 1])
    cta_split_num: List[int] = field(default_factory=lambda: [1, 1])
    cta_order: List[int] = field(default_factory=lambda: [1, 0])

    @property
    def num_threads(self):
        t = 1
        for x in self.threads_per_warp:
            t *= x
        for x in self.warps_per_cta:
            t *= x
        return t

    def elems_per_thread(self, shape):
        """Compute how many elements each thread handles for a given tensor shape.

        For each dimension, the number of elements per thread is:
          elems_in_dim = max(1, shape[d] // threads_in_dim) * sizePerThread[d]
        But when shape[d] == 1 (broadcast dim), elems = 1.
        """
        rank = len(shape)
        result = 1
        for d in range(rank):
            s = shape[d]
            spt = self.size_per_thread[d] if d < len(self.size_per_thread) else 1
            tpw = self.threads_per_warp[d] if d < len(self.threads_per_warp) else 1
            wpc = self.warps_per_cta[d] if d < len(self.warps_per_cta) else 1
            threads_in_dim = tpw * wpc
            if s == 1:
                result *= 1  # broadcast dimension
            else:
                elems = max(1, s // (threads_in_dim * spt)) * spt
                result *= elems
        return max(result, 1)

    def elems_per_thread_per_dim(self, shape):
        """Compute elements per thread along each dimension."""
        rank = len(shape)
        result = []
        for d in range(rank):
            spt = self.size_per_thread[d] if d < len(self.size_per_thread) else 1
            tpw = self.threads_per_warp[d] if d < len(self.threads_per_warp) else 1
            wpc = self.warps_per_cta[d] if d < len(self.warps_per_cta) else 1
            # Total threads covering this dimension
            threads_dim = tpw * wpc
            # Elements per thread = ceil(shape[d] / (threads_dim * sizePerThread)) * sizePerThread
            # But typically shape[d] == threads_dim * sizePerThread * repeat
            repeat = shape[d] // (threads_dim * spt) if threads_dim * spt > 0 else 1
            result.append(spt * max(repeat, 1))
        return result


@dataclass
class SliceLayout:
    parent: Any  # parent layout
    dim: int  # sliced dimension


@dataclass
class SharedLayout:
    vec: int = 1
    per_phase: int = 1
    max_phase: int = 1
    order: List[int] = field(default_factory=lambda: [1, 0])
    has_leading_offset: bool = False


@dataclass
class MMALayout:
    version_major: int
    version_minor: int
    warps_per_cta: List[int]
    ctas_per_cga: List[int] = field(default_factory=lambda: [1, 1])
    cta_split_num: List[int] = field(default_factory=lambda: [1, 1])
    cta_order: List[int] = field(default_factory=lambda: [1, 0])
    instruction_shape: List[int] = field(default_factory=lambda: [16, 8, 16])


@dataclass
class DotOperandLayout:
    op_idx: int  # 0 = A, 1 = B
    parent: Any  # parent MMA layout
    k_width: int = 0


@dataclass
class TensorType:
    shape: List[int]
    element_type: str
    layout: Any = None  # BlockedLayout, MMALayout, etc.


@dataclass
class PtrType:
    element_type: str
    address_space: int = 1  # 1 = global


@dataclass
class MemDescType:
    shape: List[int]
    element_type: str
    layout: Any = None


@dataclass
class IRValue:
    name: str
    type_str: str  # raw type string
    parsed_type: Any = None  # parsed type


@dataclass
class IROperation:
    name: str  # e.g., "tt.load", "arith.addf"
    results: List[IRValue]
    operands: List[str]  # SSA value names
    attributes: Dict[str, str]
    type_str: str  # raw type annotation string
    raw_text: str  # original MLIR text
    regions: List['IRRegion'] = field(default_factory=list)


@dataclass
class IRBlock:
    args: List[IRValue] = field(default_factory=list)
    ops: List[IROperation] = field(default_factory=list)


@dataclass
class IRRegion:
    blocks: List[IRBlock] = field(default_factory=list)


@dataclass
class IRFunction:
    name: str
    args: List[IRValue]
    body: IRRegion = None
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class IRModule:
    attrs: Dict[str, Any] = field(default_factory=dict)
    functions: List[IRFunction] = field(default_factory=list)
    layout_aliases: Dict[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    threads_per_warp: int = 32
    num_ctas: int = 1
    shared_size: int = 0


# ---------------------------------------------------------------------------
# MLIR Text Parser
# ---------------------------------------------------------------------------

class MLIRTextParser:
    """Parses MLIR text representation of TTGIR modules."""

    def parse_module(self, text: str) -> IRModule:
        module = IRModule()
        # Parse layout aliases
        module.layout_aliases = self._parse_layout_aliases(text)
        # Parse module attributes
        module.attrs = self._parse_module_attrs(text)
        module.num_warps = int(module.attrs.get('ttg.num-warps', 4))
        module.threads_per_warp = int(module.attrs.get('ttg.threads-per-warp', 32))
        module.num_ctas = int(module.attrs.get('ttg.num-ctas', 1))
        shared = module.attrs.get('ttg.shared', None)
        if shared is not None:
            module.shared_size = int(shared)
        # Parse functions
        module.functions = self._parse_functions(text, module.layout_aliases)
        return module

    def _parse_layout_aliases(self, text: str) -> Dict[str, Any]:
        aliases = {}
        # Match #name = #ttg.XXX<{...}> for all layout types
        for m in re.finditer(r'(#\w+)\s*=\s*(#ttg\.[\w_]+)<\{([^}]*(?:\{[^}]*\}[^}]*)*)\}>', text):
            alias_name = m.group(1)
            layout_type = m.group(2)
            layout_body = m.group(3)
            layout = self._parse_layout(layout_type, layout_body)
            if layout is not None:
                aliases[alias_name] = layout

        # Parse #smem = #ttg.shared_memory (no params)
        for m in re.finditer(r'(#\w+)\s*=\s*#ttg\.shared_memory\b', text):
            aliases[m.group(1)] = SharedLayout()

        # Parse #ttg.slice<{dim = N, parent = #xxx}>
        for m in re.finditer(r'#ttg\.slice<\{dim\s*=\s*(\d+),\s*parent\s*=\s*(#\w+)\}>', text):
            # These are inline, not aliased - handled during type parsing
            pass

        return aliases

    def _parse_layout(self, layout_type: str, body: str) -> Any:
        if 'nvmma_shared' in layout_type:
            return self._parse_nvmma_shared_layout(body)
        elif 'shared' in layout_type and 'nvmma' not in layout_type:
            return self._parse_shared_layout(body)
        elif 'blocked' in layout_type:
            return self._parse_blocked_layout(body)
        elif 'nvidia_mma' in layout_type:
            return self._parse_mma_layout(body)
        elif 'slice' in layout_type:
            return self._parse_slice_layout(body)
        return None

    def _parse_nvmma_shared_layout(self, body: str) -> SharedLayout:
        """Parse #ttg.nvmma_shared<{swizzlingByteWidth=64, transposed=false, elementBitWidth=16}>"""
        sbw = re.search(r'swizzlingByteWidth\s*=\s*(\d+)', body)
        trans = re.search(r'transposed\s*=\s*(\w+)', body)
        ebw = re.search(r'elementBitWidth\s*=\s*(\d+)', body)
        return SharedLayout(
            vec=int(sbw.group(1)) if sbw else 64,
            per_phase=1,
            max_phase=1,
            order=[1, 0],
            has_leading_offset=trans and trans.group(1) == 'true',
        )

    def _parse_blocked_layout(self, body: str) -> BlockedLayout:
        spt = self._extract_int_list(body, 'sizePerThread')
        tpw = self._extract_int_list(body, 'threadsPerWarp')
        wpc = self._extract_int_list(body, 'warpsPerCTA')
        order = self._extract_int_list(body, 'order')
        return BlockedLayout(
            size_per_thread=spt or [1],
            threads_per_warp=tpw or [32],
            warps_per_cta=wpc or [4],
            order=order or [0],
        )

    def _parse_mma_layout(self, body: str) -> MMALayout:
        version_match = re.search(r'versionMajor\s*=\s*(\d+)', body)
        version_minor_match = re.search(r'versionMinor\s*=\s*(\d+)', body)
        wpc = self._extract_int_list(body, 'warpsPerCTA')
        instr = self._extract_int_list(body, 'instrShape')
        return MMALayout(
            version_major=int(version_match.group(1)) if version_match else 2,
            version_minor=int(version_minor_match.group(1)) if version_minor_match else 0,
            warps_per_cta=wpc or [4, 1],
            instruction_shape=instr or [16, 8, 16],
        )

    def _parse_slice_layout(self, body: str) -> SliceLayout:
        dim_match = re.search(r'dim\s*=\s*(\d+)', body)
        # parent parsing would need recursive lookup - simplified for POC
        return SliceLayout(parent=None, dim=int(dim_match.group(1)) if dim_match else 0)

    def _parse_shared_layout(self, body: str) -> SharedLayout:
        vec_m = re.search(r'vec\s*=\s*(\d+)', body)
        pp_m = re.search(r'perPhase\s*=\s*(\d+)', body)
        mp_m = re.search(r'maxPhase\s*=\s*(\d+)', body)
        order = self._extract_int_list(body, 'order')
        hlo_m = re.search(r'hasLeadingOffset\s*=\s*(\w+)', body)
        return SharedLayout(
            vec=int(vec_m.group(1)) if vec_m else 1,
            per_phase=int(pp_m.group(1)) if pp_m else 1,
            max_phase=int(mp_m.group(1)) if mp_m else 1,
            order=order or [1, 0],
            has_leading_offset=hlo_m and hlo_m.group(1) == 'true' if hlo_m else False,
        )

    def _extract_int_list(self, text: str, key: str) -> Optional[List[int]]:
        m = re.search(rf'{key}\s*=\s*\[([^\]]*)\]', text)
        if m:
            vals = m.group(1).strip()
            if vals:
                return [int(x.strip()) for x in vals.split(',')]
        return None

    def _parse_module_attrs(self, text: str) -> Dict[str, Any]:
        attrs = {}
        m = re.search(r'module\s+attributes\s*\{([^}]+)\}', text)
        if m:
            attr_text = m.group(1)
            for am in re.finditer(r'"([^"]+)"\s*=\s*(\d+)', attr_text):
                attrs[am.group(1)] = am.group(2)
        return attrs

    def _parse_functions(self, text: str, layout_aliases: Dict) -> List[IRFunction]:
        functions = []
        # Find tt.func blocks - handle loc annotations in args
        func_pattern = r'tt\.func\s+(?:public\s+)?@(\w+)\(([^)]*(?:\([^)]*(?:\([^)]*\))?[^)]*\)[^)]*)*)\)(?:\s*attributes\s*\{[^}]*\})?\s*\{'
        for m in re.finditer(func_pattern, text):
            func_name = m.group(1)
            args_text = m.group(2)
            func_start = m.end()
            # Find matching closing brace
            func_body_text = self._find_matching_brace(text, func_start - 1)
            args = self._parse_func_args(args_text, layout_aliases)
            func = IRFunction(name=func_name, args=args)
            func.body = self._parse_region(func_body_text, layout_aliases)
            functions.append(func)
        return functions

    def _find_matching_brace(self, text: str, start: int) -> str:
        """Find text between matching { } starting at position of opening brace."""
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start + 1:i]
            i += 1
        return text[start + 1:]

    def _parse_func_args(self, args_text: str, layout_aliases: Dict) -> List[IRValue]:
        args = []
        # Split on comma but not within nested < > or { }
        parts = self._split_args(args_text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Strip loc info from args
            part = re.sub(r'\s*loc\([^)]*(?:\([^)]*\))?[^)]*\)', '', part).strip()
            # Strip trailing parenthesis that may leak from function signature parsing
            part = part.rstrip(')')
            part = part.strip()
            # Match %argN: type {attrs}
            m = re.match(r'(%[\w-]+)\s*:\s*(.+?)(?:\s*\{[^}]*\})?\s*$', part)
            if m:
                name = m.group(1)
                type_str = m.group(2).strip()
                args.append(IRValue(name=name, type_str=type_str))
        return args

    def _split_args(self, text: str) -> List[str]:
        """Split on commas, respecting nested brackets."""
        parts = []
        depth = 0
        current = []
        for ch in text:
            if ch in '(<{':
                depth += 1
                current.append(ch)
            elif ch in ')>}':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _parse_region(self, text: str, layout_aliases: Dict) -> IRRegion:
        """Parse a region (content between { })."""
        region = IRRegion()
        # For simplicity, treat entire region as one block
        block = IRBlock()
        lines = text.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('//') or line.startswith('#loc'):
                i += 1
                continue
            # Strip trailing loc info
            line = re.sub(r'\s*loc\([^)]*(?:\([^)]*\))?[^)]*\)\s*$', '', line).strip()
            if not line:
                i += 1
                continue
            # Block argument header (^bb0(%arg: type):)
            if line.startswith('^'):
                block_args_m = re.findall(r'(%[\w-]+)\s*:\s*([^,)]+)', line)
                for name, type_str in block_args_m:
                    block.args.append(IRValue(name=name, type_str=type_str.strip()))
                i += 1
                continue
            # Check if this line starts a region-containing op
            op, consumed = self._parse_op_with_regions(lines, i, layout_aliases)
            if op:
                block.ops.append(op)
                i += consumed
            else:
                op = self._parse_single_op(line, layout_aliases)
                if op:
                    block.ops.append(op)
                i += 1
        region.blocks.append(block)
        return region

    def _parse_op_with_regions(self, lines: List[str], start: int, layout_aliases: Dict) -> Tuple[Optional[IROperation], int]:
        """Try to parse an op that contains regions (scf.for, scf.if, tt.reduce, etc.)."""
        line = lines[start].strip()

        # Check for scf.for
        if 'scf.for' in line:
            return self._parse_scf_for(lines, start, layout_aliases)
        # Check for scf.if
        if 'scf.if' in line:
            return self._parse_scf_if(lines, start, layout_aliases)
        # Check for tt.reduce or "tt.reduce"
        if 'tt.reduce' in line and '{' in self._collect_until_matching(lines, start):
            return self._parse_reduce(lines, start, layout_aliases)

        return None, 0

    def _collect_until_matching(self, lines: List[str], start: int) -> str:
        """Collect lines until braces are balanced."""
        result = []
        depth = 0
        for i in range(start, min(start + 5, len(lines))):
            result.append(lines[i])
            depth += lines[i].count('{') - lines[i].count('}')
            if depth <= 0 and i > start:
                break
        return '\n'.join(result)

    def _parse_scf_for(self, lines: List[str], start: int, layout_aliases: Dict) -> Tuple[IROperation, int]:
        """Parse scf.for with its body region."""
        # Collect all lines of the for loop
        collected, consumed = self._collect_region_lines(lines, start)
        full_text = '\n'.join(collected)

        # Parse the header
        header = lines[start].strip()
        results = []
        operands = []
        attrs = {}

        # Extract results
        result_m = re.match(r'((?:%[\w-]+(?:,\s*%[\w-]+)*)\s*(?::\d+)?\s*=\s*)?scf\.for\s+(.*)', header)
        if result_m and result_m.group(1):
            result_names = re.findall(r'%[\w-]+(?::\d+)?', result_m.group(1))
            for rn in result_names:
                results.append(IRValue(name=rn, type_str=''))

        # Extract loop bounds: %iv = %lb to %ub step %step
        bounds_m = re.search(r'(%[\w-]+)\s*=\s*(%[\w-]+(?::\d+)?|%c\w+)\s+to\s+(%[\w-]+(?::\d+)?|%c\w+)\s+step\s+(%[\w-]+(?::\d+)?|%c\w+)', header)
        if bounds_m:
            attrs['iv'] = bounds_m.group(1)
            attrs['lb'] = bounds_m.group(2)
            attrs['ub'] = bounds_m.group(3)
            attrs['step'] = bounds_m.group(4)

        # Extract iter_args
        iter_args_m = re.search(r'iter_args\(([^)]*)\)', header)
        if iter_args_m:
            attrs['iter_args'] = iter_args_m.group(1)

        # Parse body region
        body_start = full_text.find('{')
        if body_start >= 0:
            body_text = self._find_matching_brace(full_text, body_start)
            region = self._parse_region(body_text, layout_aliases)
        else:
            region = IRRegion()

        # Extract result types from the end
        type_m = re.search(r'->\s*\(([^)]+)\)\s*$', full_text.rstrip().rstrip('}').strip())
        if not type_m:
            type_m = re.search(r'->\s*(\S+)\s*\{', header)

        op = IROperation(
            name='scf.for',
            results=results,
            operands=operands,
            attributes=attrs,
            type_str=type_m.group(1) if type_m else '',
            raw_text=full_text,
            regions=[region],
        )
        return op, consumed

    def _parse_scf_if(self, lines: List[str], start: int, layout_aliases: Dict) -> Tuple[IROperation, int]:
        """Parse scf.if with then/else regions."""
        collected, consumed = self._collect_region_lines(lines, start)
        full_text = '\n'.join(collected)

        header = lines[start].strip()
        results = []
        attrs = {}

        # Extract condition operand
        cond_m = re.search(r'scf\.if\s+(%[\w-]+(?::\d+)?)', header)
        if cond_m:
            attrs['cond'] = cond_m.group(1)

        # Extract results
        result_m = re.match(r'((?:%[\w-]+(?:,\s*%[\w-]+)*)\s*=\s*)?scf\.if', header)
        if result_m and result_m.group(1):
            result_names = re.findall(r'%[\w-]+', result_m.group(1))
            for rn in result_names:
                results.append(IRValue(name=rn, type_str=''))

        # Split into then/else regions
        regions = []
        # Find first { for then region
        then_start = full_text.find('{')
        if then_start >= 0:
            then_text = self._find_matching_brace(full_text, then_start)
            regions.append(self._parse_region(then_text, layout_aliases))

            # Check for else
            rest = full_text[then_start + len(then_text) + 2:]
            else_m = re.search(r'else\s*\{', rest)
            if else_m:
                else_start = rest.find('{', else_m.start())
                else_text = self._find_matching_brace(rest, else_start)
                regions.append(self._parse_region(else_text, layout_aliases))

        op = IROperation(
            name='scf.if',
            results=results,
            operands=[],
            attributes=attrs,
            type_str='',
            raw_text=full_text,
            regions=regions,
        )
        return op, consumed

    def _parse_reduce(self, lines: List[str], start: int, layout_aliases: Dict) -> Tuple[IROperation, int]:
        """Parse tt.reduce with its combiner region."""
        collected, consumed = self._collect_region_lines(lines, start)
        full_text = '\n'.join(collected)

        header = lines[start].strip()
        results = []
        operands = []
        attrs = {}

        # Extract result
        result_m = re.match(r'(%[\w-]+(?::\d+)?)\s*=\s*"?tt\.reduce"?\s*\(([^)]*)\)', header)
        if result_m:
            results.append(IRValue(name=result_m.group(1), type_str=''))
            operands = [x.strip() for x in result_m.group(2).split(',') if x.strip()]

        # Extract axis
        axis_m = re.search(r'axis\s*=\s*(\d+)', full_text)
        if axis_m:
            attrs['axis'] = axis_m.group(1)

        # Parse combiner region — find the opening ({ not the <{ from attributes
        # The combiner region starts with "({" and ends with "})"
        region = IRRegion()
        region_start = full_text.find('({')
        if region_start >= 0:
            body_text = self._find_matching_brace(full_text, region_start + 1)  # +1 to skip (
            region = self._parse_region(body_text, layout_aliases)

        op = IROperation(
            name='tt.reduce',
            results=results,
            operands=operands,
            attributes=attrs,
            type_str='',
            raw_text=full_text,
            regions=[region],
        )
        return op, consumed

    def _collect_region_lines(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect lines for an op with nested regions, tracking brace depth."""
        collected = []
        depth = 0
        i = start
        while i < len(lines):
            line = lines[i]
            collected.append(line)
            depth += line.count('{') - line.count('}')
            i += 1
            if depth <= 0 and i > start + 1:
                break
        return collected, i - start

    def _parse_single_op(self, line: str, layout_aliases: Dict) -> Optional[IROperation]:
        """Parse a single-line MLIR operation."""
        line = line.strip()
        if not line:
            return None
        # Strip location info: loc(#locN) or loc("name"(#locN))
        line = re.sub(r'\s*loc\([^)]*(?:\([^)]*\))?[^)]*\)\s*$', '', line)
        line = line.strip()
        if not line:
            return None
        # Skip terminators we handle specially
        if line.startswith('tt.return'):
            return IROperation(name='tt.return', results=[], operands=[],
                             attributes={}, type_str='', raw_text=line)

        results = []
        operands = []
        attrs = {}
        op_name = ''

        # Pattern: %result = op_name operands {attrs} : types
        # or: %result:N = op_name ...
        # or: op_name operands : types  (no result)

        # Extract results
        result_match = re.match(r'((?:%[\w.\-]+(?::\d+)?(?:,\s*%[\w.\-]+(?::\d+)?)*)\s*=\s*)', line)
        rest = line
        if result_match:
            result_text = result_match.group(1).rstrip('= ').strip()
            result_names = re.findall(r'%[\w.\-]+(?::\d+)?', result_text)
            for rn in result_names:
                results.append(IRValue(name=rn, type_str=''))
            rest = line[result_match.end():]

        # Extract op name
        op_match = re.match(r'"?([a-zA-Z_][\w.]*)"?\s*', rest)
        if op_match:
            op_name = op_match.group(1)
            rest = rest[op_match.end():]
        else:
            return None

        # Extract operands (% prefixed values, including #N suffix for multi-results)
        operand_text = rest.split('{')[0]
        # Remove the type annotation part (after last unbalanced :)
        depth_op = 0
        last_colon_op = -1
        for ci, ch in enumerate(operand_text):
            if ch in '<({':
                depth_op += 1
            elif ch in '>)}':
                depth_op -= 1
            elif ch == ':' and depth_op == 0:
                last_colon_op = ci
        if last_colon_op >= 0:
            operand_text = operand_text[:last_colon_op]
        operands = re.findall(r'%[\w.\-]+(?:[:#]\d+)?', operand_text)

        # Extract attributes
        attr_match = re.search(r'\{([^}]*)\}', rest)
        if attr_match:
            attr_text = attr_match.group(1)
            for am in re.finditer(r'(\w+)\s*=\s*("?[^,}"]+(?:"[^"]*")?)', attr_text):
                attrs[am.group(1)] = am.group(2).strip()

        # Extract type string
        # For ops with '->', the result type is after the arrow
        # For ops without '->', the type is after the last ':'
        type_str = ''
        if '->' in line:
            arrow_match = re.search(r'->\s*(.+?)$', line)
            if arrow_match:
                type_str = arrow_match.group(1).strip()
        else:
            # Find the LAST colon that's followed by a type
            # Skip colons inside nested < >
            last_colon = -1
            depth = 0
            for i, ch in enumerate(line):
                if ch in '<({':
                    depth += 1
                elif ch in '>)}':
                    depth -= 1
                elif ch == ':' and depth == 0:
                    last_colon = i
            if last_colon >= 0:
                type_str = line[last_colon+1:].strip()

        return IROperation(
            name=op_name,
            results=results,
            operands=operands,
            attributes=attrs,
            type_str=type_str,
            raw_text=line,
        )

    def parse_type_str(self, type_str: str, layout_aliases: Dict) -> Any:
        """Parse a type string into a structured type."""
        type_str = type_str.strip()
        if type_str.startswith('tensor<'):
            return self._parse_tensor_type(type_str, layout_aliases)
        elif type_str.startswith('!tt.ptr<'):
            inner = re.search(r'!tt\.ptr<(.+)>', type_str)
            if inner:
                return PtrType(element_type=inner.group(1))
        return type_str  # scalar type

    def _parse_tensor_type(self, type_str: str, layout_aliases: Dict) -> TensorType:
        # tensor<128x64xf16, #blocked>
        m = re.match(r'tensor<(.+)>', type_str)
        if not m:
            return TensorType(shape=[], element_type='f32')
        inner = m.group(1)
        # Split shape and layout
        parts = inner.rsplit(',', 1)
        shape_and_type = parts[0].strip()
        layout = None
        if len(parts) > 1:
            layout_ref = parts[1].strip()
            if layout_ref in layout_aliases:
                layout = layout_aliases[layout_ref]

        # Parse shape and element type
        dims = shape_and_type.split('x')
        element_type = dims[-1]
        shape = [int(d) for d in dims[:-1]]

        return TensorType(shape=shape, element_type=element_type, layout=layout)


# ---------------------------------------------------------------------------
# Layout Computer
# ---------------------------------------------------------------------------

class LayoutComputer:
    """Computes per-thread element indices from layout encodings."""

    def __init__(self, threads_per_warp: int = 32):
        self.threads_per_warp = threads_per_warp

    def get_thread_index_exprs(self, layout: BlockedLayout, shape: List[int],
                                dim: int = 0) -> Tuple[str, int]:
        """Generate CUDA expression for computing element indices for a blocked layout.

        Returns (index_expression_template, num_elements_per_thread).
        The template uses {tid}, {warp_id}, {lane_id}, {elem} as placeholders.
        """
        rank = len(shape)
        if rank == 1:
            return self._get_1d_index_expr(layout, shape[0])
        elif rank == 2:
            return self._get_2d_index_expr(layout, shape)
        else:
            # Fallback for higher dims
            return self._get_1d_index_expr(layout, math.prod(shape))

    def _get_1d_index_expr(self, layout: BlockedLayout, size: int) -> Tuple[str, int]:
        spt = layout.size_per_thread[0]
        tpw = layout.threads_per_warp[0]
        wpc = layout.warps_per_cta[0]
        total_threads = tpw * wpc
        elems_per_thread = size // total_threads

        # Thread indexing: each thread handles `spt` contiguous elements,
        # then jumps by total_threads * spt
        # idx = lane_in_dim * spt + (elem % spt) + (elem / spt) * total_threads * spt
        # where lane_in_dim = tid % tpw + (tid / threads_per_warp) * tpw  (if single-dim warp layout)

        # Simplified for 1D blocked:
        # thread_offset = (lane_id + warp_id * tpw) * spt
        # elem_offset = (elem_inner) + (elem_outer * total_threads * spt)
        # where elem_inner = elem % spt, elem_outer = elem / spt
        if elems_per_thread == spt:
            # Simple case: each thread handles exactly spt contiguous elements
            expr = "({{lane_id}} + {{warp_id}} * {tpw}) * {spt} + {{elem}}".format(
                tpw=tpw, spt=spt)
        else:
            # Multiple blocks of spt
            n_blocks = elems_per_thread // spt
            expr = ("({{lane_id}} + {{warp_id}} * {tpw}) * {spt} + "
                    "({{elem}} % {spt}) + ({{elem}} / {spt}) * {stride}").format(
                tpw=tpw, spt=spt, stride=total_threads * spt)
        return expr, elems_per_thread

    def _get_2d_index_expr(self, layout: BlockedLayout, shape: List[int]) -> Tuple[str, int]:
        spt0 = layout.size_per_thread[0] if len(layout.size_per_thread) > 0 else 1
        spt1 = layout.size_per_thread[1] if len(layout.size_per_thread) > 1 else 1
        tpw0 = layout.threads_per_warp[0] if len(layout.threads_per_warp) > 0 else 1
        tpw1 = layout.threads_per_warp[1] if len(layout.threads_per_warp) > 1 else 1
        wpc0 = layout.warps_per_cta[0] if len(layout.warps_per_cta) > 0 else 1
        wpc1 = layout.warps_per_cta[1] if len(layout.warps_per_cta) > 1 else 1

        total_threads_0 = tpw0 * wpc0
        total_threads_1 = tpw1 * wpc1
        elems_per_thread_0 = shape[0] // total_threads_0 if total_threads_0 > 0 else shape[0]
        elems_per_thread_1 = shape[1] // total_threads_1 if total_threads_1 > 0 else shape[1]
        total_elems = elems_per_thread_0 * elems_per_thread_1

        return "2d", total_elems


# ---------------------------------------------------------------------------
# CUDA Type Mapper
# ---------------------------------------------------------------------------

def _split_args(text: str) -> List[str]:
    """Split on commas, respecting nested brackets."""
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch in '(<{':
            depth += 1
            current.append(ch)
        elif ch in ')>}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


MLIR_TO_CUDA_TYPE = {
    'f32': 'float',
    'f16': '__half',
    'bf16': '__nv_bfloat16',
    'f64': 'double',
    'i1': 'bool',
    'i8': 'int8_t',
    'i16': 'int16_t',
    'i32': 'int',
    'i64': 'int64_t',
    'u8': 'uint8_t',
    'u16': 'uint16_t',
    'u32': 'unsigned int',
    'u64': 'uint64_t',
}


def mlir_type_to_cuda(type_str: str) -> str:
    """Convert MLIR scalar type name to CUDA type name."""
    type_str = type_str.strip()
    # Handle pointer types
    if type_str.startswith('!tt.ptr<'):
        inner = re.search(r'!tt\.ptr<(.+?)(?:,\s*\d+)?\s*>', type_str)
        if inner:
            return mlir_type_to_cuda(inner.group(1)) + '*'
        return 'void*'
    return MLIR_TO_CUDA_TYPE.get(type_str, type_str)


def mlir_type_to_format_spec(type_str: str) -> str:
    """Get printf format specifier for a type."""
    if type_str in ('f32', 'f64', 'f16', 'bf16'):
        return '%f'
    elif type_str in ('i32', 'i16', 'i8'):
        return '%d'
    elif type_str in ('i64',):
        return '%ld'
    elif type_str in ('i1',):
        return '%d'
    return '%d'


# ---------------------------------------------------------------------------
# CUDA Code Generator
# ---------------------------------------------------------------------------

class CUDACodeGen:
    """Generates CUDA C++ code from parsed TTGIR IR."""

    def __init__(self, module: IRModule, capability: int):
        self.module = module
        self.capability = capability
        self.indent_level = 0
        self.lines: List[str] = []
        self.var_counter = 0
        self.ssa_to_var: Dict[str, str] = {}  # %name -> cuda_var_name
        self.ssa_to_type: Dict[str, str] = {}  # %name -> element type string
        self.ssa_is_tensor: Dict[str, bool] = {}  # %name -> whether it's a tensor
        self.ssa_is_ptr_tensor: Dict[str, bool] = {}  # %name -> whether it's a tensor of pointers
        self.ssa_tensor_info: Dict[str, Tuple[List[int], Any]] = {}  # %name -> (shape, layout)
        self.shared_mem_offset = 0
        self.shared_allocs: Dict[str, int] = {}  # var -> offset
        self.kernel_name = ""
        self.layout_computer = LayoutComputer(module.threads_per_warp)

    def generate(self) -> str:
        """Generate complete CUDA source code."""
        self._emit_headers()
        for func in self.module.functions:
            self._emit_function(func)
        return '\n'.join(self.lines)

    def _emit(self, line: str):
        indent = '    ' * self.indent_level
        self.lines.append(indent + line)

    def _emit_blank(self):
        self.lines.append('')

    def _emit_headers(self):
        self._emit('#include <cuda.h>')
        self._emit('#include <cuda_fp16.h>')
        self._emit('#include <cuda_bf16.h>')
        self._emit('#include <stdint.h>')
        self._emit('#include <float.h>')
        self._emit('#include <math.h>')
        self._emit_blank()
        self._emit('// Helper macros')
        self._emit('#ifndef __CUDA_ARCH__')
        self._emit('#define __CUDA_ARCH__ %d0' % (self.capability // 10))
        self._emit('#endif')
        self._emit_blank()
        # Helper for warp shuffle
        self._emit('__device__ __forceinline__ float warp_reduce_sum(float val) {')
        self._emit('    #pragma unroll')
        self._emit('    for (int offset = 16; offset > 0; offset /= 2)')
        self._emit('        val += __shfl_xor_sync(0xffffffff, val, offset);')
        self._emit('    return val;')
        self._emit('}')
        self._emit_blank()
        self._emit('__device__ __forceinline__ float warp_reduce_max(float val) {')
        self._emit('    #pragma unroll')
        self._emit('    for (int offset = 16; offset > 0; offset /= 2)')
        self._emit('        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));')
        self._emit('    return val;')
        self._emit('}')
        self._emit_blank()

    def _new_var(self, prefix: str = 'v') -> str:
        self.var_counter += 1
        return f'{prefix}{self.var_counter}'

    def _get_var(self, ssa_name: str) -> str:
        """Get or create a CUDA variable name for an SSA value."""
        if ssa_name in self.ssa_to_var:
            return self.ssa_to_var[ssa_name]
        # Try without the :N suffix (for result names like %accumulator:3)
        base = ssa_name.split(':')[0]
        if base in self.ssa_to_var:
            return self.ssa_to_var[base]
        # Create new variable
        var = self._new_var()
        self.ssa_to_var[ssa_name] = var
        return var

    def _register_var(self, ssa_name: str, cuda_var: str, elem_type: str = '',
                      is_tensor: bool = False, shape: List[int] = None, layout: Any = None):
        self.ssa_to_var[ssa_name] = cuda_var
        if elem_type:
            self.ssa_to_type[ssa_name] = elem_type
        self.ssa_is_tensor[ssa_name] = is_tensor
        if shape is not None:
            self.ssa_tensor_info[ssa_name] = (shape, layout)

    def _get_elem_type(self, ssa_name: str) -> str:
        return self.ssa_to_type.get(ssa_name, 'float')

    def _emit_function(self, func: IRFunction):
        self.kernel_name = func.name
        num_threads = self.module.num_warps * self.module.threads_per_warp

        # Build argument list
        args = []
        for arg in func.args:
            cuda_type = self._convert_arg_type(arg.type_str)
            var_name = self._get_var(arg.name)
            self._register_var(arg.name, var_name, self._get_scalar_type(arg.type_str))
            args.append(f'{cuda_type} {var_name}')

        # Emit kernel signature
        self._emit(f'extern "C" __global__ void __launch_bounds__({num_threads})')
        self._emit(f'{func.name}({", ".join(args)}) {{')
        self.indent_level += 1

        # Emit thread indexing
        self._emit('// Thread indexing')
        self._emit('const int tid = threadIdx.x;')
        self._emit('const int warp_id = tid / 32;')
        self._emit('const int lane_id = tid % 32;')
        self._emit_blank()

        # Emit shared memory (always declare for potential convert_layout use)
        self._emit(f'extern __shared__ char shared_mem[];')
        self._emit_blank()

        # Emit function body
        if func.body and func.body.blocks:
            for block in func.body.blocks:
                self._emit_block(block)

        self.indent_level -= 1
        self._emit('}')

    def _convert_arg_type(self, type_str: str) -> str:
        """Convert MLIR argument type to CUDA type."""
        type_str = type_str.strip()
        if type_str.startswith('!tt.ptr<'):
            inner = re.search(r'!tt\.ptr<(.+?)(?:,\s*\d+)?\s*>', type_str)
            if inner:
                return mlir_type_to_cuda(inner.group(1)) + '*'
        if type_str.startswith('!tt.tensordesc<'):
            # TMA tensor descriptor — 128-byte CUtensorMap, passed as __grid_constant__
            return 'const __grid_constant__ CUtensorMap'
        return mlir_type_to_cuda(type_str)

    def _get_scalar_type(self, type_str: str) -> str:
        """Extract scalar element type from a type string."""
        type_str = type_str.strip()
        if type_str.startswith('!tt.ptr<'):
            inner = re.search(r'!tt\.ptr<(.+?)(?:,\s*\d+)?\s*>', type_str)
            return inner.group(1) if inner else 'f32'
        if type_str.startswith('tensor<'):
            parts = type_str[7:].rstrip('>').split('x')
            # Get element type (last part before optional layout)
            for p in reversed(parts):
                p = p.split(',')[0].strip()
                if p in MLIR_TO_CUDA_TYPE:
                    return p
        return type_str

    def _emit_block(self, block: IRBlock):
        for op in block.ops:
            self._emit_op(op)

    def _emit_op(self, op: IROperation):
        """Dispatch to the appropriate emitter for each op type."""
        name = op.name

        if name == 'tt.return':
            self._emit('return;')
        elif name == 'arith.constant':
            self._emit_arith_constant(op)
        elif name == 'tt.get_program_id':
            self._emit_get_program_id(op)
        elif name == 'tt.get_num_programs':
            self._emit_get_num_programs(op)
        elif name == 'tt.make_range':
            self._emit_make_range(op)
        elif name == 'tt.splat':
            self._emit_splat(op)
        elif name == 'tt.broadcast':
            self._emit_broadcast(op)
        elif name == 'tt.expand_dims':
            self._emit_expand_dims(op)
        elif name == 'tt.load':
            self._emit_load(op)
        elif name == 'tt.store':
            self._emit_store(op)
        elif name == 'tt.dot':
            self._emit_dot(op)
        elif name == 'tt.reduce':
            self._emit_reduce(op)
        elif name == 'tt.trans':
            self._emit_trans(op)
        elif name == 'tt.addptr':
            self._emit_addptr(op)
        elif name == 'scf.for':
            self._emit_scf_for(op)
        elif name == 'scf.if':
            self._emit_scf_if(op)
        elif name == 'scf.yield':
            self._emit_scf_yield(op)
        elif name == 'tt.reduce.return':
            pass  # handled by reduce emitter
        elif name.startswith('arith.') or name.startswith('arith '):
            self._emit_arith(op)
        elif name.startswith('math.'):
            self._emit_math(op)
        elif name == 'ttg.convert_layout':
            self._emit_convert_layout(op)
        elif name == 'ttg.local_alloc':
            self._emit_local_alloc(op)
        elif name == 'ttg.local_store':
            self._emit_local_store(op)
        elif name == 'ttg.local_load':
            self._emit_local_load(op)
        elif name in ('ttg.memdesc_subview', 'ttg.memdesc_index'):
            self._emit_memdesc_subview(op)
        elif name == 'ttg.local_dealloc':
            self._emit(f'// local_dealloc (no-op in CUDA)')
        elif name == 'gpu.barrier':
            self._emit('__syncthreads();')
        # NVGPUIR ops (sm90a features)
        elif name == 'ttng.warp_group_dot':
            self._emit_warp_group_dot(op)
        elif name == 'ttng.warp_group_dot_wait':
            self._emit_warp_group_dot_wait(op)
        elif name == 'ttng.fence_async_shared':
            self._emit_fence_async_shared(op)
        elif name == 'ttng.init_barrier':
            self._emit_init_barrier(op)
        elif name == 'ttng.wait_barrier':
            self._emit_wait_barrier(op)
        elif name == 'ttng.arrive_barrier':
            self._emit_arrive_barrier(op)
        elif name == 'ttng.barrier_expect':
            self._emit_barrier_expect(op)
        elif name == 'ttng.inval_barrier':
            bar_ops = self._extract_barrier_ops(op) if hasattr(self, '_extract_barrier_ops') else op.operands
            if bar_ops:
                bar_var = self._get_var(bar_ops[0])
                self._emit(f'if (threadIdx.x == 0)')
                self._emit(f'    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"((unsigned)__cvta_generic_to_shared({bar_var})));')
            else:
                self._emit('// inval_barrier (no barrier operand found)')
        elif name == 'ttng.async_tma_copy_global_to_local':
            self._emit_tma_copy_g2l(op)
        elif name == 'ttng.async_tma_copy_local_to_global':
            self._emit_tma_copy_l2g(op)
        elif name in ('ttng.tma_store_wait', 'ttng.async_tma_store_wait'):
            pendings = re.search(r'pendings\s*=\s*(\d+)', op.raw_text)
            n = pendings.group(1) if pendings else '0'
            self._emit(f'asm volatile("cp.async.bulk.wait_group.read {n};");')
        elif name == 'ttng.async_copy_mbarrier_arrive':
            self._emit('// async_copy_mbarrier_arrive (handled by TMA)')
        elif name == 'tt.mulhiui':
            self._emit_mulhiui(op)
        elif name == 'tt.bitcast':
            self._emit_tt_bitcast(op)
        elif name == 'tt.print':
            self._emit_print(op)
        elif name == 'tt.assert':
            pass  # skip asserts in generated code
        elif name == 'tt.extern_elementwise':
            self._emit_extern_elementwise(op)
        elif name == 'tt.atomic_rmw':
            self._emit_atomic_rmw(op)
        elif name == 'tt.atomic_cas':
            self._emit_atomic_cas(op)
        else:
            self._emit(f'// TODO: unhandled op: {name}')
            self._emit(f'// {op.raw_text.strip()}')

    # -----------------------------------------------------------------------
    # Individual Op Emitters
    # -----------------------------------------------------------------------

    def _emit_arith_constant(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        raw = op.raw_text.strip()

        # Detect tensor constant (dense)
        if 'dense' in raw:
            # dense<0.0> : tensor<...>
            val_m = re.search(r'dense<([^>]+)>', raw)
            val = val_m.group(1) if val_m else '0'
            # Get tensor type info
            tt = self._extract_tensor_type(op.type_str)
            if tt:
                cuda_type = mlir_type_to_cuda(tt.element_type)
                n_elems = self._get_elems_for_tensor(tt)
                var = self._new_var('c')
                self._register_var(result.name, var, tt.element_type,
                                 is_tensor=True, shape=tt.shape, layout=tt.layout)
                val_cuda = self._format_constant_value(val, tt.element_type)
                self._emit(f'{cuda_type} {var}[{n_elems}];')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {val_cuda};')
            return

        # Scalar constant
        val_m = re.search(r'constant\s+(.+?)\s*:\s*(\S+)', raw)
        if val_m:
            val = val_m.group(1).strip()
            type_str = val_m.group(2).strip()
            cuda_type = mlir_type_to_cuda(type_str)
            var = self._new_var('c')
            # Clean up value representation
            val_cuda = self._format_constant_value(val, type_str)
            self._emit(f'const {cuda_type} {var} = {val_cuda};')
            self._register_var(result.name, var, type_str)

    def _format_constant_value(self, val: str, type_str: str) -> str:
        """Format a constant value for CUDA."""
        val = val.strip()
        if val == 'true':
            return 'true'
        if val == 'false':
            return 'false'
        # Handle special float hex values first
        if type_str in ('f32', 'f16', 'bf16', 'f64'):
            if val == '0xFF800000' or val == '0xff800000':
                return '-INFINITY'
            if val == '0x7F800000' or val == '0x7f800000':
                return 'INFINITY'
            if val == '0x7FC00000' or val == '0x7fc00000':
                return 'NAN'
            if 'inf' in val.lower():
                return 'INFINITY' if '-' not in val else '-INFINITY'
        # Handle generic hex float literals
        if val.startswith('0x') and type_str in ('f32', 'f16', 'bf16', 'f64'):
            return val
            try:
                fval = float(val)
                if type_str == 'f32':
                    return f'{fval}f'
                return str(fval)
            except ValueError:
                pass
        # Handle negative zero
        if val == '-0.000000e+00' or val == '0.000000e+00':
            if type_str == 'f32':
                return '0.0f'
            return '0.0'
        return val

    def _emit_get_program_id(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        var = self._new_var('pid')
        raw = op.raw_text
        if 'x' in raw.split('get_program_id')[1].split(':')[0]:
            axis = 'blockIdx.x'
        elif 'y' in raw.split('get_program_id')[1].split(':')[0]:
            axis = 'blockIdx.y'
        else:
            axis = 'blockIdx.z'
        self._emit(f'const int {var} = {axis};')
        self._register_var(result.name, var, 'i32')

    def _emit_get_num_programs(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        var = self._new_var('npg')
        raw = op.raw_text
        if 'x' in raw.split('get_num_programs')[1].split(':')[0]:
            axis = 'gridDim.x'
        elif 'y' in raw.split('get_num_programs')[1].split(':')[0]:
            axis = 'gridDim.y'
        else:
            axis = 'gridDim.z'
        self._emit(f'const int {var} = {axis};')
        self._register_var(result.name, var, 'i32')

    def _emit_make_range(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        raw = op.raw_text
        start_m = re.search(r'start\s*=\s*(\d+)', raw)
        end_m = re.search(r'end\s*=\s*(\d+)', raw)
        start = int(start_m.group(1)) if start_m else 0
        end = int(end_m.group(1)) if end_m else 0
        size = end - start

        # Get layout from type
        tt = self._extract_tensor_type(op.type_str)
        if tt and isinstance(tt.layout, BlockedLayout):
            layout = tt.layout
            n_elems = layout.elems_per_thread([size])
            var = self._new_var('rng')
            self._register_var(result.name, var, 'i32',
                             is_tensor=True, shape=[size], layout=layout)

            spt = layout.size_per_thread[0]
            tpw = layout.threads_per_warp[0]
            wpc = layout.warps_per_cta[0]
            total_threads = tpw * wpc
            stride = total_threads * spt

            self._emit(f'int {var}[{n_elems}];')
            self._emit('{')
            self.indent_level += 1
            self._emit(f'int _base = (lane_id % {tpw} + warp_id * {tpw}) * {spt} + {start};')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {{')
            self.indent_level += 1
            if n_elems == spt:
                self._emit(f'{var}[_i] = _base + _i;')
            else:
                self._emit(f'{var}[_i] = _base + (_i % {spt}) + (_i / {spt}) * {stride};')
            self.indent_level -= 1
            self._emit('}')
            self.indent_level -= 1
            self._emit('}')
        else:
            # Fallback: simple sequential range per thread
            n_elems = size // (self.module.num_warps * self.module.threads_per_warp)
            n_elems = max(n_elems, 1)
            var = self._new_var('rng')
            self._register_var(result.name, var, 'i32', is_tensor=True, shape=[size])
            self._emit(f'int {var}[{n_elems}];')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = tid * {n_elems} + _i + {start};')

    def _emit_splat(self, op: IROperation):
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])

        # Check if this is a pointer splat (tensor of pointers)
        is_ptr_splat = '!tt.ptr' in op.type_str and 'tensor' in op.type_str
        tt = self._extract_tensor_type(op.type_str)

        if is_ptr_splat and tt:
            # Pointer splat: scalar ptr -> tensor of ptrs
            n_elems = self._get_elems_for_tensor(tt)
            # Use the full element type (which includes ptr)
            elem_cuda_type = mlir_type_to_cuda(tt.element_type)
            var = self._new_var('spl')
            self._register_var(result.name, var, tt.element_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)
            self.ssa_is_ptr_tensor[result.name] = True
            self._emit(f'{elem_cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var};')
        elif tt:
            n_elems = self._get_elems_for_tensor(tt)
            cuda_type = mlir_type_to_cuda(tt.element_type)
            var = self._new_var('spl')
            self._register_var(result.name, var, tt.element_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)
            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var};')
        else:
            # Scalar splat
            var = self._new_var('spl')
            self._register_var(result.name, var, src_type)
            cuda_type = mlir_type_to_cuda(src_type)
            self._emit(f'{cuda_type} {var} = {src_var};')

    def _emit_broadcast(self, op: IROperation):
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src = op.operands[0]
        src_var = self._get_var(src)

        tt = self._extract_tensor_type(op.type_str)
        if tt:
            n_elems = self._get_elems_for_tensor(tt)
            src_is_tensor = self.ssa_is_tensor.get(src, False)
            src_is_ptr = self.ssa_is_ptr_tensor.get(src, False)
            is_ptr_tensor = src_is_ptr or '!tt.ptr' in (tt.element_type or '')

            elem_type = tt.element_type
            cuda_type = mlir_type_to_cuda(elem_type)

            if src_is_tensor:
                src_n = self._get_num_elems(src)
                src_n = max(src_n, 1)
                var = self._new_var('bcast')
                self._register_var(result.name, var, elem_type,
                                 is_tensor=True, shape=tt.shape, layout=tt.layout)
                if is_ptr_tensor:
                    self.ssa_is_ptr_tensor[result.name] = True
                self._emit(f'{cuda_type} {var}[{n_elems}];')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var}[_i % {src_n}];')
            else:
                var = self._new_var('bcast')
                self._register_var(result.name, var, elem_type,
                                 is_tensor=True, shape=tt.shape, layout=tt.layout)
                if is_ptr_tensor:
                    self.ssa_is_ptr_tensor[result.name] = True
                self._emit(f'{cuda_type} {var}[{n_elems}];')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var};')

    def _emit_expand_dims(self, op: IROperation):
        """Expand dims - adds a size-1 dimension. Per-thread data count may change."""
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])
        src_is_tensor = self.ssa_is_tensor.get(op.operands[0], False)
        src_n = self._get_num_elems(op.operands[0]) if src_is_tensor else 1

        tt = self._extract_tensor_type(op.type_str)
        if tt:
            n_elems = self._get_elems_for_tensor(tt)
            n_elems = max(n_elems, src_n)  # Expand dims shouldn't reduce element count
            cuda_type = mlir_type_to_cuda(tt.element_type if tt.element_type in MLIR_TO_CUDA_TYPE else src_type)
            var = self._new_var('exp')
            self._register_var(result.name, var, tt.element_type if tt.element_type in MLIR_TO_CUDA_TYPE else src_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)
            if src_is_tensor:
                # Alias or copy
                if n_elems == src_n:
                    self._emit(f'{cuda_type}* {var} = {src_var}; // expand_dims (alias)')
                else:
                    self._emit(f'{cuda_type} {var}[{n_elems}];')
                    self._emit(f'#pragma unroll')
                    self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var}[_i % {src_n}];')
            else:
                self._emit(f'{cuda_type} {var}[{n_elems}];')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {src_var};')

    def _emit_addptr(self, op: IROperation):
        if not op.results or len(op.operands) < 2:
            return
        result = op.results[0]
        ptr_var = self._get_var(op.operands[0])
        offset_var = self._get_var(op.operands[1])

        ptr_is_tensor = self.ssa_is_tensor.get(op.operands[0], False)
        offset_is_tensor = self.ssa_is_tensor.get(op.operands[1], False)
        ptr_type = self._get_elem_type(op.operands[0])

        if ptr_is_tensor or offset_is_tensor:
            n_elems = max(
                self._get_num_elems(op.operands[0]) if ptr_is_tensor else 1,
                self._get_num_elems(op.operands[1]) if offset_is_tensor else 1,
            )
            tt = self._extract_tensor_type(op.type_str)
            var = self._new_var('ptr')
            cuda_type = mlir_type_to_cuda(ptr_type)
            self._register_var(result.name, var, ptr_type,
                             is_tensor=True,
                             shape=tt.shape if tt else [],
                             layout=tt.layout if tt else None)
            self.ssa_is_ptr_tensor[result.name] = True
            # For pointer tensors, each element is already a pointer
            # cuda_type might be '__half*' already, so don't add extra *
            decl_type = cuda_type if cuda_type.endswith('*') else cuda_type + '*'
            self._emit(f'{decl_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            p = f'{ptr_var}[_i]' if ptr_is_tensor else ptr_var
            o = f'{offset_var}[_i]' if offset_is_tensor else offset_var
            self._emit(f'    {var}[_i] = {p} + {o};')
        else:
            var = self._new_var('ptr')
            cuda_type = mlir_type_to_cuda(ptr_type)
            self._register_var(result.name, var, ptr_type)
            decl_type = cuda_type if cuda_type.endswith('*') else cuda_type + '*'
            self._emit(f'{decl_type} {var} = {ptr_var} + {offset_var};')

    def _emit_load(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        ptr_operand = op.operands[0] if op.operands else None

        # For tt.load, the result element type is the POINTEE type of the ptr
        src_elem_type = self._get_elem_type(ptr_operand) if ptr_operand else 'f32'
        # Strip pointer wrapper if present
        if src_elem_type.startswith('!tt.ptr<'):
            inner_m = re.search(r'!tt\.ptr<(.+?)(?:,\s*\d+)?\s*>', src_elem_type)
            src_elem_type = inner_m.group(1) if inner_m else 'f32'
        # Get tensor shape/layout from the type annotation
        tt = self._extract_tensor_type(op.type_str)
        # Override element type to be the pointee type (not ptr type)
        if tt:
            tt = TensorType(shape=tt.shape, element_type=src_elem_type, layout=tt.layout)

        if tt and ptr_operand:
            n_elems = self._get_elems_for_tensor(tt)
            cuda_type = mlir_type_to_cuda(tt.element_type)
            var = self._new_var('ld')
            self._register_var(result.name, var, tt.element_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)

            ptr_var = self._get_var(ptr_operand)
            has_mask = len(op.operands) >= 2
            has_other = len(op.operands) >= 3
            mask_var = self._get_var(op.operands[1]) if has_mask else None
            other_var = self._get_var(op.operands[2]) if has_other else None

            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {{')
            self.indent_level += 1
            if has_mask:
                self._emit(f'if ({mask_var}[_i]) {{')
                self.indent_level += 1
                self._emit(f'{var}[_i] = *{ptr_var}[_i];')
                self.indent_level -= 1
                if has_other:
                    other_is_tensor = self.ssa_is_tensor.get(op.operands[2], False)
                    self._emit(f'}} else {{')
                    self.indent_level += 1
                    if other_is_tensor:
                        self._emit(f'{var}[_i] = {other_var}[_i];')
                    else:
                        self._emit(f'{var}[_i] = {other_var};')
                    self.indent_level -= 1
                    self._emit('}')
                else:
                    self._emit(f'}} else {{')
                    self.indent_level += 1
                    self._emit(f'{var}[_i] = 0;')
                    self.indent_level -= 1
                    self._emit('}')
            else:
                self._emit(f'{var}[_i] = *{ptr_var}[_i];')
            self.indent_level -= 1
            self._emit('}')
        else:
            # Scalar load
            var = self._new_var('ld')
            if ptr_operand:
                ptr_var = self._get_var(ptr_operand)
                elem_type = self._get_elem_type(ptr_operand)
                cuda_type = mlir_type_to_cuda(elem_type)
                self._register_var(result.name, var, elem_type)
                self._emit(f'{cuda_type} {var} = *{ptr_var};')

    def _emit_store(self, op: IROperation):
        if len(op.operands) < 2:
            return
        ptr_operand = op.operands[0]
        val_operand = op.operands[1]
        ptr_var = self._get_var(ptr_operand)
        val_var = self._get_var(val_operand)
        has_mask = len(op.operands) >= 3
        mask_var = self._get_var(op.operands[2]) if has_mask else None

        is_tensor = self.ssa_is_tensor.get(val_operand, False)
        ptr_is_tensor = self.ssa_is_tensor.get(ptr_operand, False)
        if is_tensor and ptr_is_tensor:
            n_elems = self._get_num_elems(val_operand)
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {{')
            self.indent_level += 1
            if has_mask:
                self._emit(f'if ({mask_var}[_i])')
                self._emit(f'    *{ptr_var}[_i] = {val_var}[_i];')
            else:
                self._emit(f'*{ptr_var}[_i] = {val_var}[_i];')
            self.indent_level -= 1
            self._emit('}')
        else:
            if has_mask:
                self._emit(f'if ({mask_var}) *{ptr_var} = {val_var};')
            else:
                self._emit(f'*{ptr_var} = {val_var};')

    def _emit_dot(self, op: IROperation):
        """Emit matrix multiplication. For POC, use simple FMA accumulation."""
        if not op.results or len(op.operands) < 3:
            return
        result = op.results[0]
        a_var = self._get_var(op.operands[0])
        b_var = self._get_var(op.operands[1])
        acc_var = self._get_var(op.operands[2])
        acc_type = self._get_elem_type(op.operands[2])

        tt = self._extract_tensor_type(op.type_str)
        if tt:
            n_elems = self._get_elems_for_tensor(tt)
            cuda_type = mlir_type_to_cuda(tt.element_type)
            var = self._new_var('dot')
            self._register_var(result.name, var, tt.element_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)

            # For POC: copy accumulator and add naive dot product
            # TODO: Replace with MMA PTX inline asm for performance
            self._emit(f'// tt.dot — matrix multiply accumulate')
            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {var}[_i] = {acc_var}[_i];')
            self._emit(f'// TODO: Implement proper MMA for performance')
            self._emit(f'// Current: copy accumulator (dot product requires MMA PTX inline asm)')
        else:
            var = self._new_var('dot')
            self._register_var(result.name, var, acc_type)
            self._emit(f'{mlir_type_to_cuda(acc_type)} {var} = {acc_var}; // TODO: dot')

    def _emit_reduce(self, op: IROperation):
        """Emit reduction operation."""
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])
        n_elems = self._get_num_elems(op.operands[0])

        axis = int(op.attributes.get('axis', '0'))

        # Determine reduction type from combiner region
        reduce_op = 'add'  # default
        if op.regions:
            combiner = op.regions[0]
            if combiner.blocks:
                for cop in combiner.blocks[0].ops:
                    if 'addf' in cop.name or 'addi' in cop.name:
                        reduce_op = 'add'
                    elif 'maxnumf' in cop.name or 'maxf' in cop.name or 'maximumf' in cop.name:
                        reduce_op = 'max'
                    elif 'minnumf' in cop.name or 'minf' in cop.name or 'minimumf' in cop.name:
                        reduce_op = 'min'
                    elif 'mulf' in cop.name:
                        reduce_op = 'mul'
                    elif 'xori' in cop.name:
                        reduce_op = 'xor'

        cuda_type = mlir_type_to_cuda(src_type)
        var = self._new_var('red')

        # Check if result is scalar or tensor
        result_tt = self._extract_tensor_type(op.type_str)
        if result_tt:
            result_n = self._get_elems_for_tensor(result_tt)
            self._register_var(result.name, var, src_type,
                             is_tensor=True, shape=result_tt.shape, layout=result_tt.layout)
            self._emit(f'{cuda_type} {var}[{result_n}];')
            self._emit(f'// Reduction along axis={axis} (TODO: full implementation)')
            self._emit(f'for (int _i = 0; _i < {result_n}; _i++) {var}[_i] = {src_var}[_i];')
        else:
            self._register_var(result.name, var, src_type)
            # Scalar result - full reduction
            identity = self._get_reduce_identity(reduce_op, src_type)
            self._emit(f'{cuda_type} {var} = {identity};')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            self._emit(f'    {var} = {self._get_reduce_expr(reduce_op, var, f"{src_var}[_i]", src_type)};')
            # Warp reduction
            self._emit(f'// Warp-level reduction')
            if reduce_op == 'add':
                self._emit(f'{var} = warp_reduce_sum({var});')
            elif reduce_op == 'max':
                self._emit(f'{var} = warp_reduce_max({var});')
            else:
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _off = 16; _off > 0; _off /= 2)')
                self._emit(f'    {var} = {self._get_reduce_expr(reduce_op, var, f"__shfl_xor_sync(0xffffffff, {var}, _off)", src_type)};')

            # Cross-warp reduction if needed
            if self.module.num_warps > 1:
                self._emit(f'// Cross-warp reduction')
                self._emit(f'{{')
                self.indent_level += 1
                self._emit(f'__shared__ {cuda_type} _warp_buf[{self.module.num_warps}];')
                self._emit(f'if (lane_id == 0) _warp_buf[warp_id] = {var};')
                self._emit(f'__syncthreads();')
                self._emit(f'if (warp_id == 0) {{')
                self.indent_level += 1
                self._emit(f'{var} = (lane_id < {self.module.num_warps}) ? _warp_buf[lane_id] : {identity};')
                if reduce_op == 'add':
                    self._emit(f'{var} = warp_reduce_sum({var});')
                elif reduce_op == 'max':
                    self._emit(f'{var} = warp_reduce_max({var});')
                else:
                    self._emit(f'for (int _off = 16; _off > 0; _off /= 2)')
                    self._emit(f'    {var} = {self._get_reduce_expr(reduce_op, var, f"__shfl_xor_sync(0xffffffff, {var}, _off)", src_type)};')
                self.indent_level -= 1
                self._emit(f'}}')
                self._emit(f'// Broadcast result from warp 0 to all warps')
                self._emit(f'_warp_buf[0] = {var};')
                self._emit(f'__syncthreads();')
                self._emit(f'{var} = _warp_buf[0];')
                self.indent_level -= 1
                self._emit(f'}}')

    def _get_reduce_identity(self, reduce_op: str, type_str: str) -> str:
        if reduce_op == 'add':
            return '0.0f' if type_str in ('f32', 'f16', 'bf16', 'f64') else '0'
        elif reduce_op == 'max':
            return '-INFINITY' if type_str in ('f32', 'f16', 'bf16', 'f64') else 'INT_MIN'
        elif reduce_op == 'min':
            return 'INFINITY' if type_str in ('f32', 'f16', 'bf16', 'f64') else 'INT_MAX'
        elif reduce_op == 'mul':
            return '1.0f' if type_str in ('f32', 'f16', 'bf16', 'f64') else '1'
        return '0'

    def _get_reduce_expr(self, reduce_op: str, a: str, b: str, type_str: str) -> str:
        if reduce_op == 'add':
            return f'({a} + {b})'
        elif reduce_op == 'max':
            if type_str in ('f32', 'f16', 'bf16', 'f64'):
                return f'fmaxf({a}, {b})'
            return f'max({a}, {b})'
        elif reduce_op == 'min':
            if type_str in ('f32', 'f16', 'bf16', 'f64'):
                return f'fminf({a}, {b})'
            return f'min({a}, {b})'
        elif reduce_op == 'mul':
            return f'({a} * {b})'
        elif reduce_op == 'xor':
            return f'({a} ^ {b})'
        return f'({a} + {b})'

    def _emit_trans(self, op: IROperation):
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])
        # Trans is a metadata change, data doesn't move in per-thread view
        var = self._new_var('tr')
        n_elems = self._get_num_elems(op.operands[0])
        tt = self._extract_tensor_type(op.type_str)
        self._register_var(result.name, var, src_type,
                         is_tensor=True,
                         shape=tt.shape if tt else [],
                         layout=tt.layout if tt else None)
        cuda_type = mlir_type_to_cuda(src_type)
        self._emit(f'{cuda_type}* {var} = {src_var}; // transpose (metadata only)')

    def _emit_arith(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        name = op.name

        is_tensor = any(self.ssa_is_tensor.get(o, False) for o in op.operands)
        src_type = self._get_elem_type(op.operands[0]) if op.operands else 'f32'

        # For select, the result type comes from the value operands (not the condition)
        if 'select' in name and len(op.operands) >= 3:
            src_type = self._get_elem_type(op.operands[1])

        # If type annotation has tensor type, extract element type from there
        # For select: type annotation is "tensor<Nxi1>, tensor<Nxf32>" — use the LAST tensor type
        type_str_for_check = op.type_str
        if 'select' in name and ',' in op.type_str:
            # Use the value type (last tensor in the type list)
            parts = op.type_str.rsplit('tensor<', 1)
            if len(parts) > 1:
                type_str_for_check = 'tensor<' + parts[-1]
        tt_check = self._extract_tensor_type(type_str_for_check)
        if tt_check and tt_check.element_type in MLIR_TO_CUDA_TYPE:
            src_type = tt_check.element_type

        # Determine result type
        result_type = src_type
        if 'extf' in name or 'sitofp' in name or 'uitofp' in name:
            result_type = self._extract_result_type(op)
        elif 'truncf' in name or 'fptosi' in name or 'fptoui' in name:
            result_type = self._extract_result_type(op)
        elif 'extsi' in name or 'extui' in name or 'trunci' in name:
            result_type = self._extract_result_type(op)
        elif 'bitcast' in name:
            result_type = self._extract_result_type(op)

        cuda_type = mlir_type_to_cuda(result_type)

        # For cmp ops, extract predicate from raw_text
        arith_name = name
        if 'cmpi' in name or 'cmpf' in name:
            # raw text: "arith.cmpi slt, %a, %b" -> extract predicate
            raw_after_op = op.raw_text.split('cmpi')[1] if 'cmpi' in op.raw_text else op.raw_text.split('cmpf')[1]
            pred_m = re.match(r'\s+(\w+)', raw_after_op)
            if pred_m:
                arith_name = name + ' ' + pred_m.group(1)

        if is_tensor:
            n_elems = max(self._get_num_elems(o) for o in op.operands if self.ssa_is_tensor.get(o, False))
            tt = self._extract_tensor_type(op.type_str)
            var = self._new_var('a')
            # cmpi result is always i1/bool tensor
            actual_result_type = result_type
            if 'cmp' in name:
                actual_result_type = 'i1'
            self._register_var(result.name, var, actual_result_type, is_tensor=True,
                             shape=tt.shape if tt else [], layout=tt.layout if tt else None)
            result_cuda_type = mlir_type_to_cuda(actual_result_type) if 'cmp' in name else cuda_type
            self._emit(f'{result_cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            expr = self._arith_expr(arith_name, op.operands, '_i', is_tensor=True, result_type=result_type)
            self._emit(f'    {var}[_i] = {expr};')
        else:
            var = self._new_var('a')
            actual_result_type = 'i1' if 'cmp' in name else result_type
            self._register_var(result.name, var, actual_result_type)
            expr = self._arith_expr(arith_name, op.operands, None, is_tensor=False, result_type=result_type)
            result_cuda_type = mlir_type_to_cuda(actual_result_type) if 'cmp' in name else cuda_type
            self._emit(f'const {result_cuda_type} {var} = {expr};')

    def _arith_expr(self, name: str, operands: List[str], idx: Optional[str],
                    is_tensor: bool = False, result_type: str = '') -> str:
        """Generate arithmetic expression."""
        def v(i):
            var = self._get_var(operands[i])
            if is_tensor and self.ssa_is_tensor.get(operands[i], False):
                return f'{var}[{idx}]'
            return var

        result_cuda = mlir_type_to_cuda(result_type) if result_type else ''

        op = name.replace('arith.', '')

        # Binary ops
        if op in ('addi', 'addf'):
            return f'({v(0)} + {v(1)})'
        elif op in ('subi', 'subf'):
            return f'({v(0)} - {v(1)})'
        elif op in ('muli', 'mulf'):
            return f'({v(0)} * {v(1)})'
        elif op in ('divsi',):
            return f'({v(0)} / {v(1)})'
        elif op in ('divui',):
            cast = '(uint64_t)' if result_type in ('i64',) else '(unsigned)'
            return f'({cast}{v(0)} / {cast}{v(1)})'
        elif op in ('divf',):
            return f'({v(0)} / {v(1)})'
        elif op in ('remsi',):
            return f'({v(0)} % {v(1)})'
        elif op in ('remui',):
            return f'((unsigned){v(0)} % (unsigned){v(1)})'
        elif op in ('andi',):
            return f'({v(0)} & {v(1)})'
        elif op in ('ori',):
            return f'({v(0)} | {v(1)})'
        elif op in ('xori',):
            return f'({v(0)} ^ {v(1)})'
        elif op in ('shli',):
            return f'({v(0)} << {v(1)})'
        elif op in ('shrsi',):
            return f'({v(0)} >> {v(1)})'
        elif op in ('shrui',):
            cast = '(uint64_t)' if result_type in ('i64',) else '(unsigned)'
            return f'({cast}{v(0)} >> {v(1)})'
        # Comparison ops - predicate is in the op name like "cmpi slt"
        elif op.startswith('cmpf') or op.startswith('cmpi'):
            parts = op.split()
            pred = parts[1] if len(parts) >= 2 else 'eq'
            return self._cmp_expr(pred, v(0), v(1))
        # Unary/conversion ops
        elif op in ('negf',):
            return f'(-{v(0)})'
        elif op in ('select',):
            return f'({v(0)} ? {v(1)} : {v(2)})'
        elif op in ('extf',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('truncf',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('extsi',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('extui',):
            return f'(({result_cuda})(unsigned){v(0)})'
        elif op in ('trunci',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('sitofp',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('uitofp',):
            return f'(({result_cuda})(unsigned){v(0)})'
        elif op in ('fptosi',):
            return f'(({result_cuda}){v(0)})'
        elif op in ('fptoui',):
            return f'(({result_cuda})(unsigned){v(0)})'
        elif op in ('bitcast',):
            return f'__int_as_float({v(0)})' if result_type == 'f32' else f'__float_as_int({v(0)})'
        elif op in ('maxf', 'maximumf'):
            return f'fmaxf({v(0)}, {v(1)})'
        elif op in ('minf', 'minimumf'):
            return f'fminf({v(0)}, {v(1)})'
        elif op in ('maxsi',):
            return f'max({v(0)}, {v(1)})'
        elif op in ('minsi',):
            return f'min({v(0)}, {v(1)})'
        elif op in ('maxui',):
            return f'max((unsigned){v(0)}, (unsigned){v(1)})'
        elif op in ('minui',):
            return f'min((unsigned){v(0)}, (unsigned){v(1)})'
        elif op in ('index_cast', 'index_castui'):
            return f'(({result_cuda}){v(0)})'
        else:
            return f'/* TODO: arith.{op} */ {v(0)}'

    def _extract_cmp_pred_from_raw(self, full_op_name: str) -> str:
        """Extract comparison predicate from the full op text.

        The raw text looks like: 'arith.cmpi slt, %a, %b : type'
        We extract 'slt' from the op raw text.
        """
        # The predicate is passed as part of the operation name in MLIR text
        # e.g., "arith.cmpi slt" or "arith.cmpf olt"
        parts = full_op_name.split()
        if len(parts) >= 2:
            return parts[1].rstrip(',')
        return 'eq'

    def _cmp_expr(self, pred: str, a: str, b: str) -> str:
        cmp_map = {
            'eq': '==', 'ne': '!=',
            'slt': '<', 'sle': '<=', 'sgt': '>', 'sge': '>=',
            'ult': '<', 'ule': '<=', 'ugt': '>', 'uge': '>=',
            'oeq': '==', 'one': '!=', 'ogt': '>', 'oge': '>=',
            'olt': '<', 'ole': '<=',
            'ueq': '==', 'une': '!=',
        }
        op = cmp_map.get(pred, '==')
        return f'({a} {op} {b})'

    def _emit_math(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        name = op.name.replace('math.', '')
        is_tensor = any(self.ssa_is_tensor.get(o, False) for o in op.operands)
        src_type = self._get_elem_type(op.operands[0]) if op.operands else 'f32'
        cuda_type = mlir_type_to_cuda(src_type)

        # Math function mapping
        math_map = {
            'exp': 'expf', 'exp2': 'exp2f', 'log': 'logf', 'log2': 'log2f',
            'sqrt': 'sqrtf', 'rsqrt': 'rsqrtf',
            'sin': 'sinf', 'cos': 'cosf', 'tanh': 'tanhf',
            'absf': 'fabsf', 'abs': 'fabsf',
            'ceil': 'ceilf', 'floor': 'floorf',
            'fma': 'fmaf', 'erf': 'erff', 'powf': 'powf',
            'round': 'roundf', 'roundeven': 'rintf',
            'trunc': 'truncf',
        }
        func = math_map.get(name, name + 'f')

        if is_tensor:
            n_elems = max(self._get_num_elems(o) for o in op.operands if self.ssa_is_tensor.get(o, False))
            tt = self._extract_tensor_type(op.type_str)
            var = self._new_var('m')
            self._register_var(result.name, var, src_type, is_tensor=True,
                             shape=tt.shape if tt else [], layout=tt.layout if tt else None)
            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            args = []
            for o in op.operands:
                v = self._get_var(o)
                if self.ssa_is_tensor.get(o, False):
                    args.append(f'{v}[_i]')
                else:
                    args.append(v)
            self._emit(f'    {var}[_i] = {func}({", ".join(args)});')
        else:
            var = self._new_var('m')
            self._register_var(result.name, var, src_type)
            args = [self._get_var(o) for o in op.operands]
            self._emit(f'const {cuda_type} {var} = {func}({", ".join(args)});')

    def _emit_scf_for(self, op: IROperation):
        iv = op.attributes.get('iv', '%iv')
        lb = op.attributes.get('lb', '')
        ub = op.attributes.get('ub', '')
        step = op.attributes.get('step', '')

        lb_var = self._get_var(lb) if lb else '0'
        ub_var = self._get_var(ub) if ub else '0'
        step_var = self._get_var(step) if step else '1'
        iv_var = self._new_var('iv')
        self._register_var(iv, iv_var, 'i32')

        # Handle iter_args
        iter_args_str = op.attributes.get('iter_args', '')
        if iter_args_str:
            # Parse iter_args(%name = %init)
            iter_pairs = re.findall(r'(%[\w-]+)\s*=\s*(%[\w-]+(?::\d+)?)', iter_args_str)
            for (iter_name, init_name) in iter_pairs:
                init_var = self._get_var(init_name)
                init_type = self._get_elem_type(init_name)
                is_tensor = self.ssa_is_tensor.get(init_name, False)
                if is_tensor:
                    n_elems = self._get_num_elems(init_name)
                    is_ptr = self.ssa_is_ptr_tensor.get(init_name, False)
                    if is_ptr:
                        # For pointer tensors, get the full type including ptr
                        stored_type = self.ssa_to_type.get(init_name, init_type)
                        cuda_type = mlir_type_to_cuda(stored_type)
                        if not cuda_type.endswith('*'):
                            cuda_type += '*'
                    else:
                        cuda_type = mlir_type_to_cuda(init_type)
                    iter_var = self._new_var('iter')
                    self._register_var(iter_name, iter_var, self.ssa_to_type.get(init_name, init_type),
                                     is_tensor=True,
                                     shape=self.ssa_tensor_info.get(init_name, ([],))[0],
                                     layout=self.ssa_tensor_info.get(init_name, ([], None))[1])
                    if is_ptr:
                        self.ssa_is_ptr_tensor[iter_name] = True
                    self._emit(f'{cuda_type} {iter_var}[{n_elems}];')
                    self._emit(f'for (int _t = 0; _t < {n_elems}; _t++) {iter_var}[_t] = {init_var}[_t];')
                else:
                    cuda_type = mlir_type_to_cuda(init_type)
                    iter_var = self._new_var('iter')
                    self._register_var(iter_name, iter_var, init_type)
                    self._emit(f'{cuda_type} {iter_var} = {init_var};')

            # Register result variables
            # For multi-result like %accumulator:3, register %accumulator#0, #1, #2
            # For single-result like %_mean, also register %_mean directly
            for i, (iter_name, init_name) in enumerate(iter_pairs):
                iter_var = self._get_var(iter_name)
                if op.results:
                    base_name = op.results[0].name.split(':')[0]
                    # Always register %name#i
                    result_name = f'{base_name}#{i}'
                    self.ssa_to_var[result_name] = iter_var
                    self.ssa_to_type[result_name] = self._get_elem_type(iter_name)
                    self.ssa_is_tensor[result_name] = self.ssa_is_tensor.get(iter_name, False)
                    if iter_name in self.ssa_tensor_info:
                        self.ssa_tensor_info[result_name] = self.ssa_tensor_info[iter_name]
                    if self.ssa_is_ptr_tensor.get(iter_name, False):
                        self.ssa_is_ptr_tensor[result_name] = True
                    # For single-result, also register base_name directly
                    if len(iter_pairs) == 1 and ':' not in op.results[0].name:
                        self.ssa_to_var[base_name] = iter_var
                        self.ssa_to_type[base_name] = self._get_elem_type(iter_name)
                        self.ssa_is_tensor[base_name] = self.ssa_is_tensor.get(iter_name, False)
                        if iter_name in self.ssa_tensor_info:
                            self.ssa_tensor_info[base_name] = self.ssa_tensor_info[iter_name]
                        if self.ssa_is_ptr_tensor.get(iter_name, False):
                            self.ssa_is_ptr_tensor[base_name] = True

        self._emit(f'for (int {iv_var} = {lb_var}; {iv_var} < {ub_var}; {iv_var} += {step_var}) {{')
        self.indent_level += 1

        # Also register block args of the body region
        if op.regions and op.regions[0].blocks:
            block = op.regions[0].blocks[0]
            if block.args:
                # First arg is IV, rest are iter_args
                if block.args:
                    self._register_var(block.args[0].name, iv_var, 'i32')
                for i, arg in enumerate(block.args[1:]):
                    if iter_args_str:
                        iter_pairs = re.findall(r'(%[\w-]+)\s*=\s*(%[\w-]+(?::\d+)?)', iter_args_str)
                        if i < len(iter_pairs):
                            iter_var = self._get_var(iter_pairs[i][0])
                            self.ssa_to_var[arg.name] = iter_var
                            self.ssa_to_type[arg.name] = self._get_elem_type(iter_pairs[i][0])
                            self.ssa_is_tensor[arg.name] = self.ssa_is_tensor.get(iter_pairs[i][0], False)

            # Emit body ops
            for bop in block.ops:
                if bop.name == 'scf.yield':
                    self._emit_scf_yield_for(bop, iter_args_str if iter_args_str else '')
                else:
                    self._emit_op(bop)

        self.indent_level -= 1
        self._emit('}')

    def _emit_scf_yield(self, op: IROperation):
        # scf.yield is handled by the parent (for or if)
        pass

    def _emit_scf_yield_for(self, op: IROperation, iter_args_str: str):
        """Handle scf.yield inside a for loop - update iter_args."""
        if not iter_args_str:
            return
        iter_pairs = re.findall(r'(%[\w-]+)\s*=\s*(%[\w-]+(?::\d+)?)', iter_args_str)
        for i, operand in enumerate(op.operands):
            if i < len(iter_pairs):
                iter_var = self._get_var(iter_pairs[i][0])
                src_var = self._get_var(operand)
                is_tensor = self.ssa_is_tensor.get(operand, False)
                if is_tensor:
                    n_elems = self._get_num_elems(operand)
                    self._emit(f'#pragma unroll')
                    self._emit(f'for (int _t = 0; _t < {n_elems}; _t++) {iter_var}[_t] = {src_var}[_t];')
                else:
                    if iter_var != src_var:
                        self._emit(f'{iter_var} = {src_var};')

    def _emit_scf_if(self, op: IROperation):
        cond = op.attributes.get('cond', '')
        cond_var = self._get_var(cond) if cond else 'true'

        self._emit(f'if ({cond_var}) {{')
        self.indent_level += 1
        if op.regions and op.regions[0].blocks:
            for bop in op.regions[0].blocks[0].ops:
                self._emit_op(bop)
        self.indent_level -= 1
        self._emit('}')

        if len(op.regions) > 1:
            self._emit('else {')
            self.indent_level += 1
            if op.regions[1].blocks:
                for bop in op.regions[1].blocks[0].ops:
                    self._emit_op(bop)
            self.indent_level -= 1
            self._emit('}')

    def _emit_convert_layout(self, op: IROperation):
        """Layout conversion between different tensor layouts.

        For #mma -> #blocked: store accumulator to shared memory via stmatrix,
        then load with blocked layout addressing.
        For other conversions: use shared memory as intermediary.
        """
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])
        is_tensor = self.ssa_is_tensor.get(op.operands[0], False)

        tt = self._extract_tensor_type(op.type_str)
        src_info = self.ssa_tensor_info.get(op.operands[0])
        src_layout = src_info[1] if src_info else None

        if is_tensor:
            n_src_elems = self._get_num_elems(op.operands[0])
            n_dst_elems = self._get_elems_for_tensor(tt) if tt else n_src_elems
            var = self._new_var('cvt')
            cuda_type = mlir_type_to_cuda(src_type)

            # Check if this is MMA -> blocked conversion (needs stmatrix + ld.shared)
            is_mma_to_blocked = isinstance(src_layout, MMALayout) and isinstance(tt.layout, BlockedLayout) if tt else False

            if is_mma_to_blocked:
                self._register_var(result.name, var, src_type, is_tensor=True,
                                 shape=tt.shape, layout=tt.layout)
                self._emit(f'// convert_layout: #mma -> #blocked via shared memory')
                self._emit(f'{cuda_type} {var}[{n_dst_elems}];')
                self._emit(f'{{')
                self.indent_level += 1

                # Allocate temp shared memory for the conversion
                total_elems = tt.shape[0] * tt.shape[1] if len(tt.shape) >= 2 else n_src_elems
                type_bytes = {'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4}
                eb = type_bytes.get(src_type, 4)
                smem_bytes = total_elems * eb
                smem_offset = self.shared_mem_offset
                smem_offset = (smem_offset + 127) & ~127
                self.shared_mem_offset = smem_offset + smem_bytes

                self._emit(f'{cuda_type}* _cvt_smem = ({cuda_type}*)(shared_mem + {smem_offset});')

                # Store accumulator to shared memory
                # For MMA layout, use stmatrix PTX or direct store
                self._emit(f'// Store MMA registers to shared memory')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_src_elems}; _i++)')
                self._emit(f'    _cvt_smem[tid * {n_src_elems} + _i] = {src_var}[_i];')
                self._emit(f'__syncthreads();')

                # Load with blocked layout
                self._emit(f'// Load with blocked layout')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_dst_elems}; _i++)')
                self._emit(f'    {var}[_i] = _cvt_smem[tid * {n_dst_elems} + _i];')
                self._emit(f'__syncthreads();')

                self.indent_level -= 1
                self._emit(f'}}')
            else:
                # Generic conversion via pointer alias (may need shared memory)
                self._register_var(result.name, var, src_type, is_tensor=True,
                                 shape=tt.shape if tt else [],
                                 layout=tt.layout if tt else None)
                if n_src_elems == n_dst_elems:
                    self._emit(f'// convert_layout (same element count - alias)')
                    self._emit(f'{cuda_type}* {var} = {src_var};')
                else:
                    self._emit(f'// convert_layout via shared memory')
                    self._emit(f'{cuda_type} {var}[{n_dst_elems}];')
                    self._emit(f'{{')
                    self.indent_level += 1
                    total_elems = tt.shape[0] * tt.shape[1] if tt and len(tt.shape) >= 2 else max(n_src_elems, n_dst_elems) * 128
                    eb = {'f32': 4, 'f16': 2, 'bf16': 2, 'i32': 4}.get(src_type, 4)
                    smem_offset = (self.shared_mem_offset + 127) & ~127
                    self.shared_mem_offset = smem_offset + total_elems * eb
                    self._emit(f'{cuda_type}* _cvt_smem = ({cuda_type}*)(shared_mem + {smem_offset});')
                    self._emit(f'#pragma unroll')
                    self._emit(f'for (int _i = 0; _i < {n_src_elems}; _i++)')
                    self._emit(f'    _cvt_smem[tid * {n_src_elems} + _i] = {src_var}[_i];')
                    self._emit(f'__syncthreads();')
                    self._emit(f'#pragma unroll')
                    self._emit(f'for (int _i = 0; _i < {n_dst_elems}; _i++)')
                    self._emit(f'    {var}[_i] = _cvt_smem[tid * {n_dst_elems} + _i];')
                    self._emit(f'__syncthreads();')
                    self.indent_level -= 1
                    self._emit(f'}}')
        else:
            var = self._new_var('cvt')
            self._register_var(result.name, var, src_type)
            self._emit(f'auto {var} = {src_var}; // convert_layout (scalar)')

    def _emit_memdesc_subview(self, op: IROperation):
        """Emit memdesc_index/memdesc_subview: index into a multi-buffered shared memory allocation.

        TTGIR: %sub = ttg.memdesc_index %base[%idx] : memdesc<3xMxNxT> -> memdesc<MxNxT>
        CUDA: T* sub = base + idx * (M * N);
        """
        if not op.results or not op.operands:
            return
        result = op.results[0]
        base_var = self._get_var(op.operands[0])
        base_type = self._get_elem_type(op.operands[0])

        # Extract index from brackets [%idx]
        idx_match = re.search(r'\[(%[\w.\-]+(?:[:#]\d+)?)\]', op.raw_text)
        idx_var = self._get_var(idx_match.group(1)) if idx_match else '0'

        # Extract result memdesc shape to compute stride
        # e.g., memdesc<3x128x32xf16> → subview is 128x32xf16, stride = 128*32
        # Result type: memdesc<128x32xf16, ...> or memdesc<1xi64, ...>
        result_m = re.search(r'->\s*!ttg\.memdesc<([^>]+)>', op.raw_text)
        stride = 1
        elem_type = base_type
        if result_m:
            inner = result_m.group(1)
            dims = []
            remaining = inner
            while remaining:
                dm = re.match(r'(\d+)x(.*)', remaining)
                if dm:
                    dims.append(int(dm.group(1)))
                    remaining = dm.group(2)
                else:
                    t = remaining.split(',')[0].strip()
                    if t in MLIR_TO_CUDA_TYPE:
                        elem_type = t
                    break
            for d in dims:
                stride *= d

        cuda_type = mlir_type_to_cuda(elem_type)
        type_size = {'float': 4, '__half': 2, '__nv_bfloat16': 2, 'double': 8,
                     'int': 4, 'int64_t': 8, 'int8_t': 1, 'int16_t': 2, 'uint64_t': 8}
        elem_bytes = type_size.get(cuda_type, 4)

        var = self._new_var('sv')
        self._register_var(result.name, var, elem_type)
        # Compute byte offset: idx * stride * sizeof(element)
        byte_stride = stride * elem_bytes
        self._emit(f'{cuda_type}* {var} = ({cuda_type}*)((char*){base_var} + {idx_var} * {byte_stride});')

    def _emit_local_alloc(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        # Get memory descriptor type
        size = 4096  # default
        elem_type = 'f32'
        raw = op.raw_text
        # Try to extract size from type annotation or result type
        # Look in both raw text and type_str
        search_text = raw + ' ' + op.type_str
        m = re.search(r'memdesc<([^>]+)>', search_text)
        if m:
            inner = m.group(1)
            # Parse dimensions
            dim_parts = []
            remaining = inner
            while remaining:
                dm = re.match(r'(\d+)x(.*)', remaining)
                if dm:
                    dim_parts.append(int(dm.group(1)))
                    remaining = dm.group(2)
                else:
                    type_part = remaining.split(',')[0].strip()
                    if type_part in MLIR_TO_CUDA_TYPE:
                        elem_type = type_part
                    break
            if dim_parts:
                size = 1
                for s in dim_parts:
                    size *= s

        type_size = {'f32': 4, 'f16': 2, 'bf16': 2, 'f64': 8, 'i32': 4, 'i8': 1, 'i16': 2, 'i64': 8}
        byte_size = size * type_size.get(elem_type, 4)

        # Check if this is a WGMMA shared alloc (has source operand to store)
        has_source = len(op.operands) > 0

        var = self._new_var('smem')
        cuda_type = mlir_type_to_cuda(elem_type)
        offset = self.shared_mem_offset
        # Align to 128 bytes for WGMMA
        offset = (offset + 127) & ~127
        self.shared_mem_offset = offset + byte_size
        self._register_var(result.name, var, elem_type)
        self._emit(f'{cuda_type}* {var} = ({cuda_type}*)(shared_mem + {offset});')

        if has_source:
            # local_alloc with source: store the tensor to shared memory
            src_var = self._get_var(op.operands[0])
            src_is_tensor = self.ssa_is_tensor.get(op.operands[0], False)
            if src_is_tensor:
                n_elems = self._get_num_elems(op.operands[0])
                self._emit(f'// Store to shared memory for WGMMA')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
                self._emit(f'    {var}[tid * {n_elems} + _i] = {src_var}[_i];')

    def _emit_local_store(self, op: IROperation):
        if len(op.operands) < 2:
            return
        val_var = self._get_var(op.operands[0])
        dst_var = self._get_var(op.operands[1])
        is_tensor = self.ssa_is_tensor.get(op.operands[0], False)

        if is_tensor:
            n_elems = self._get_num_elems(op.operands[0])
            self._emit(f'// local_store to shared memory')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            self._emit(f'    {dst_var}[tid * {n_elems} + _i] = {val_var}[_i];')
            self._emit(f'__syncthreads();')
        else:
            self._emit(f'{dst_var}[tid] = {val_var};')

    def _emit_local_load(self, op: IROperation):
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])

        tt = self._extract_tensor_type(op.type_str)
        if tt:
            n_elems = self._get_elems_for_tensor(tt)
            cuda_type = mlir_type_to_cuda(tt.element_type)
            var = self._new_var('sld')
            self._register_var(result.name, var, tt.element_type,
                             is_tensor=True, shape=tt.shape, layout=tt.layout)
            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            self._emit(f'    {var}[_i] = {src_var}[tid * {n_elems} + _i];')

    def _emit_print(self, op: IROperation):
        self._emit('// tt.print (omitted in CUDA)')

    def _emit_extern_elementwise(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        # Extract function name from attributes
        func_name = op.attributes.get('symbol', op.attributes.get('libname', 'unknown'))
        func_name = func_name.strip('"')
        # Map libdevice __nv_ intrinsics to standard CUDA math functions
        NV_TO_CUDA = {
            '__nv_asinf': 'asinf', '__nv_acosf': 'acosf', '__nv_atanf': 'atanf',
            '__nv_atan2f': 'atan2f', '__nv_sinf': 'sinf', '__nv_cosf': 'cosf',
            '__nv_tanf': 'tanf', '__nv_sinhf': 'sinhf', '__nv_coshf': 'coshf',
            '__nv_tanhf': 'tanhf', '__nv_expf': 'expf', '__nv_exp2f': 'exp2f',
            '__nv_logf': 'logf', '__nv_log2f': 'log2f', '__nv_log10f': 'log10f',
            '__nv_sqrtf': 'sqrtf', '__nv_rsqrtf': 'rsqrtf', '__nv_cbrtf': 'cbrtf',
            '__nv_ceilf': 'ceilf', '__nv_floorf': 'floorf', '__nv_truncf': 'truncf',
            '__nv_roundf': 'roundf', '__nv_fabsf': 'fabsf', '__nv_fmodf': 'fmodf',
            '__nv_powf': 'powf', '__nv_fmaf': 'fmaf', '__nv_erff': 'erff',
            '__nv_erfcf': 'erfcf', '__nv_copysignf': 'copysignf',
            '__nv_fmaxf': 'fmaxf', '__nv_fminf': 'fminf',
            '__nv_asin': 'asin', '__nv_acos': 'acos', '__nv_atan': 'atan',
            '__nv_sin': 'sin', '__nv_cos': 'cos', '__nv_exp': 'exp',
            '__nv_log': 'log', '__nv_sqrt': 'sqrt', '__nv_fabs': 'fabs',
            '__nv_pow': 'pow', '__nv_fma': 'fma', '__nv_erf': 'erf',
        }
        func_name = NV_TO_CUDA.get(func_name, func_name)

        is_tensor = any(self.ssa_is_tensor.get(o, False) for o in op.operands)
        src_type = self._get_elem_type(op.operands[0]) if op.operands else 'f32'

        if is_tensor:
            n_elems = max(self._get_num_elems(o) for o in op.operands if self.ssa_is_tensor.get(o, False))
            tt = self._extract_tensor_type(op.type_str)
            cuda_type = mlir_type_to_cuda(tt.element_type if tt else src_type)
            var = self._new_var('ext')
            self._register_var(result.name, var, tt.element_type if tt else src_type,
                             is_tensor=True, shape=tt.shape if tt else [], layout=tt.layout if tt else None)
            self._emit(f'{cuda_type} {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++) {{')
            self.indent_level += 1
            args = []
            for o in op.operands:
                v = self._get_var(o)
                if self.ssa_is_tensor.get(o, False):
                    args.append(f'{v}[_i]')
                else:
                    args.append(v)
            self._emit(f'{var}[_i] = {func_name}({", ".join(args)});')
            self.indent_level -= 1
            self._emit('}')
        else:
            cuda_type = mlir_type_to_cuda(src_type)
            var = self._new_var('ext')
            self._register_var(result.name, var, src_type)
            args = [self._get_var(o) for o in op.operands]
            self._emit(f'{cuda_type} {var} = {func_name}({", ".join(args)});')

    def _emit_tt_bitcast(self, op: IROperation):
        """Emit tt.bitcast: reinterpret bits without changing data."""
        if not op.results or not op.operands:
            return
        result = op.results[0]
        src_var = self._get_var(op.operands[0])
        src_type = self._get_elem_type(op.operands[0])
        is_tensor = self.ssa_is_tensor.get(op.operands[0], False)

        # Extract result type from type annotation (after ->)
        result_type = src_type
        tt = self._extract_tensor_type(op.type_str)
        if tt and tt.element_type in MLIR_TO_CUDA_TYPE:
            result_type = tt.element_type

        if is_tensor:
            n_elems = self._get_num_elems(op.operands[0])
            var = self._new_var('bc')
            cuda_type = mlir_type_to_cuda(result_type)
            self._register_var(result.name, var, result_type, is_tensor=True,
                             shape=self.ssa_tensor_info.get(op.operands[0], ([],))[0],
                             layout=self.ssa_tensor_info.get(op.operands[0], ([], None))[1])
            if src_type == result_type:
                # Same type bitcast — just alias
                self._emit(f'{cuda_type}* {var} = {src_var}; // tt.bitcast (same type)')
            else:
                src_cuda = mlir_type_to_cuda(src_type)
                self._emit(f'{cuda_type} {var}[{n_elems}];')
                self._emit(f'#pragma unroll')
                self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
                # Use memcpy for proper bitcast
                self._emit(f'    {{ {src_cuda} _tmp = {src_var}[_i]; memcpy(&{var}[_i], &_tmp, sizeof(_tmp)); }}')
        else:
            var = self._new_var('bc')
            self._register_var(result.name, var, result_type)
            if src_type == result_type:
                self._emit(f'auto {var} = {src_var}; // tt.bitcast (same type)')
            else:
                cuda_type = mlir_type_to_cuda(result_type)
                src_cuda = mlir_type_to_cuda(src_type)
                self._emit(f'{cuda_type} {var}; {{ {src_cuda} _tmp = {src_var}; memcpy(&{var}, &_tmp, sizeof(_tmp)); }}')

    def _emit_mulhiui(self, op: IROperation):
        """Emit unsigned 32-bit multiply high: result = (a * b) >> 32."""
        if not op.results or len(op.operands) < 2:
            return
        result = op.results[0]
        a_var = self._get_var(op.operands[0])
        b_var = self._get_var(op.operands[1])
        is_tensor = any(self.ssa_is_tensor.get(o, False) for o in op.operands)

        if is_tensor:
            n_elems = max(self._get_num_elems(o) for o in op.operands if self.ssa_is_tensor.get(o, False))
            var = self._new_var('mhi')
            self._register_var(result.name, var, 'i32', is_tensor=True,
                             shape=self.ssa_tensor_info.get(op.operands[0], ([],))[0],
                             layout=self.ssa_tensor_info.get(op.operands[0], ([], None))[1])
            self._emit(f'int {var}[{n_elems}];')
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {n_elems}; _i++)')
            a = f'{a_var}[_i]' if self.ssa_is_tensor.get(op.operands[0], False) else a_var
            b = f'{b_var}[_i]' if self.ssa_is_tensor.get(op.operands[1], False) else b_var
            self._emit(f'    {var}[_i] = __umulhi((unsigned){a}, (unsigned){b});')
        else:
            var = self._new_var('mhi')
            self._register_var(result.name, var, 'i32')
            self._emit(f'const int {var} = __umulhi((unsigned){a_var}, (unsigned){b_var});')

    def _emit_atomic_rmw(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        self._emit(f'// TODO: atomic_rmw')
        var = self._new_var('atom')
        self._register_var(result.name, var, 'f32')
        self._emit(f'float {var} = 0.0f; // atomic_rmw placeholder')

    def _emit_atomic_cas(self, op: IROperation):
        if not op.results:
            return
        result = op.results[0]
        self._emit(f'// TODO: atomic_cas')
        var = self._new_var('cas')
        self._register_var(result.name, var, 'f32')
        self._emit(f'float {var} = 0.0f; // atomic_cas placeholder')

    # -----------------------------------------------------------------------
    # SM90a NVGPUIR Op Emitters (WGMMA, TMA, barriers)
    # -----------------------------------------------------------------------

    def _emit_warp_group_dot(self, op: IROperation):
        """Emit WGMMA (Warp Group Matrix Multiply Accumulate) using PTX inline asm.

        TTGIR: ttng.warp_group_dot %a, %b, %acc {inputPrecision=0, isAsync=true}
             : !ttg.memdesc<MxKxf16, #shared> * !ttg.memdesc<KxNxf16, #shared1>
             -> tensor<MxNxf32, #mma>
        """
        if not op.results:
            return
        result = op.results[0]
        a_operand = op.operands[0] if len(op.operands) > 0 else None
        b_operand = op.operands[1] if len(op.operands) > 1 else None
        acc_operand = op.operands[2] if len(op.operands) > 2 else None

        # Get accumulator info
        acc_var = self._get_var(acc_operand) if acc_operand else None
        acc_type = self._get_elem_type(acc_operand) if acc_operand else 'f32'
        acc_n = self._get_num_elems(acc_operand) if acc_operand else 0

        # Get shared memory operands (descriptors)
        a_var = self._get_var(a_operand) if a_operand else None
        b_var = self._get_var(b_operand) if b_operand else None

        # Parse result type for shape info
        tt = self._extract_tensor_type(op.type_str)

        # Extract MMA instruction shape from the #mma layout
        M_block = tt.shape[0] if tt and len(tt.shape) >= 2 else 128
        N_block = tt.shape[1] if tt and len(tt.shape) >= 2 else 128

        # Extract K from A's memdesc shape (from raw text)
        K_block = 32  # default
        k_match = re.search(r'memdesc<\d+x(\d+)x', op.raw_text)
        if k_match:
            K_block = int(k_match.group(1))

        # Determine element types from raw text
        a_elem = 'f16'  # default
        b_elem = 'f16'
        c_elem = 'f32'
        type_match = re.search(r'memdesc<\d+x\d+x(\w+)', op.raw_text)
        if type_match:
            a_elem = type_match.group(1)

        # WGMMA instruction shape: m64nNk16 for f16
        wgmma_m = 64
        wgmma_k = 16 if a_elem in ('f16', 'bf16') else 8
        wgmma_n = N_block  # n matches the full N block

        # Number of WGMMA instructions needed
        n_m_tiles = M_block // wgmma_m  # e.g. 128/64 = 2
        n_k_tiles = K_block // wgmma_k  # e.g. 32/16 = 2

        # Each m64nN WGMMA produces 64 output registers (for f32 output, n=128)
        # Number of output regs per wgmma = (wgmma_m / 16) * (wgmma_n / 8) * (c_bits/32)
        # For m64n128k16.f32: each thread has 64 f32 regs
        n_out_regs = (wgmma_m * wgmma_n) // (self.module.num_warps * self.module.threads_per_warp)
        # More precisely: for m64n128, 128 threads, each gets 64 values
        # Actually num_warps for WGMMA is the warp group = 4 warps = 128 threads
        n_out_regs = 64  # For m64n128: 64 f32 outputs per thread

        # Total accumulator elements per thread
        total_acc = n_out_regs * n_m_tiles

        cuda_type = mlir_type_to_cuda(c_elem)
        var = self._new_var('wgmma')
        self._register_var(result.name, var, c_elem,
                         is_tensor=True, shape=tt.shape if tt else [M_block, N_block],
                         layout=tt.layout if tt else None)

        # Emit WGMMA sequence
        self._emit(f'// WGMMA: m{wgmma_m}n{wgmma_n}k{wgmma_k}.{c_elem}.{a_elem}.{b_elem}')
        self._emit(f'// {n_m_tiles} M-tiles x {n_k_tiles} K-tiles, {n_out_regs} regs/tile')

        # Declare accumulator registers
        self._emit(f'{cuda_type} {var}[{total_acc}];')

        # Initialize from input accumulator
        if acc_var:
            self._emit(f'#pragma unroll')
            self._emit(f'for (int _i = 0; _i < {total_acc}; _i++) {var}[_i] = {acc_var}[_i];')

        # Build the PTX inline asm for WGMMA
        self._emit(f'{{')
        self.indent_level += 1

        # Compute shared memory descriptors for A and B
        self._emit(f'// Construct WGMMA shared memory descriptors')
        self._emit(f'uint32_t smem_addr_a = (unsigned)__cvta_generic_to_shared({a_var});')
        self._emit(f'uint32_t smem_addr_b = (unsigned)__cvta_generic_to_shared({b_var});')
        self._emit_blank()

        # Emit wgmma.fence
        self._emit(f'asm volatile("wgmma.fence.sync.aligned;");')
        self._emit_blank()

        # Emit WGMMA instructions for each M-tile and K-tile
        for m_tile in range(n_m_tiles):
            acc_base = m_tile * n_out_regs
            for k_tile in range(n_k_tiles):
                # Compute descriptor offsets
                # A descriptor: base + m_tile * wgmma_m * K_block * elem_bytes + k_tile * wgmma_k * elem_bytes
                a_byte_offset = m_tile * wgmma_m * K_block * 2 + k_tile * wgmma_k * 2  # f16 = 2 bytes
                b_byte_offset = k_tile * wgmma_k * N_block * 2  # B is KxN

                # Build the output register list
                out_regs = ', '.join([f'"=f"({{var}}[{acc_base + i}])'.format(var=var) for i in range(n_out_regs)])
                in_regs = ', '.join([f'"0"({var}[{acc_base + i}])' for i in range(n_out_regs)])

                # Build PTX string
                # With "+f" constraint: regs are %0..%63, desc_a=%64, desc_b=%65, scale=%66
                out_str = ', '.join([f'%{i}' for i in range(n_out_regs)])
                a_idx = n_out_regs      # desc_a
                b_idx = n_out_regs + 1  # desc_b

                ptx_instr = (f'wgmma.mma_async.sync.aligned.'
                           f'm{wgmma_m}n{wgmma_n}k{wgmma_k}.'
                           f'{c_elem}.{a_elem}.{b_elem} '
                           f'{{{out_str}}}, %{a_idx}, %{b_idx}, 1, 1, 1, 0, 1;')

                # Build the asm statement
                self._emit(f'{{')
                self.indent_level += 1
                # A desc: contiguous dim = K, stride dim = M, swizzle from A's shared layout
                elem_bytes_a = 2 if a_elem in ('f16', 'bf16') else 4
                swizzle_a = 64 if K_block * elem_bytes_a >= 64 else (32 if K_block * elem_bytes_a >= 32 else 0)
                desc_tmpl_a = self._get_wgmma_desc_template(swizzle_a, M_block)
                # B desc: contiguous dim = N, stride dim = K, swizzle from B's shared layout
                elem_bytes_b = 2 if b_elem in ('f16', 'bf16') else 4
                swizzle_b = 128 if N_block * elem_bytes_b >= 128 else (64 if N_block * elem_bytes_b >= 64 else (32 if N_block * elem_bytes_b >= 32 else 0))
                desc_tmpl_b = self._get_wgmma_desc_template(swizzle_b, K_block)
                self._emit(f'uint64_t desc_a = ((uint64_t)((smem_addr_a + {a_byte_offset}) >> 4)) | 0x{desc_tmpl_a:016X}ULL;')
                self._emit(f'uint64_t desc_b = ((uint64_t)((smem_addr_b + {b_byte_offset}) >> 4)) | 0x{desc_tmpl_b:016X}ULL;')

                # Generate the asm volatile statement
                # Use "+f" (read-write) for accumulator registers
                self._emit(f'asm volatile(')
                self.indent_level += 1
                self._emit(f'"{ptx_instr}"')

                # Output operands ("+f" = read-write for accumulator)
                out_parts = [f'"+f"({var}[{acc_base + i}])' for i in range(n_out_regs)]
                self._emit(f': {", ".join(out_parts[:8])}')
                for chunk_start in range(8, n_out_regs, 8):
                    chunk = out_parts[chunk_start:chunk_start+8]
                    self._emit(f'  , {", ".join(chunk)}')

                # Input operands (descriptors + scale)
                in_parts = ['"l"(desc_a)', '"l"(desc_b)', '"n"(1)']
                self._emit(f': {", ".join(in_parts)}')

                self.indent_level -= 1
                self._emit(f');')
                self.indent_level -= 1
                self._emit(f'}}')

        # Commit and (optionally) wait
        is_async = 'isAsync = true' in op.raw_text
        self._emit(f'asm volatile("wgmma.commit_group.sync.aligned;");')
        if not is_async:
            self._emit(f'asm volatile("wgmma.wait_group.sync.aligned 0;");')

        self.indent_level -= 1
        self._emit(f'}}')

    def _get_wgmma_desc_template(self, swizzle_bytes: int, stride_dim_size: int) -> int:
        """Compute the static part of a WGMMA shared memory descriptor.

        Args:
            swizzle_bytes: Swizzling byte width (0, 32, 64, or 128)
            stride_dim_size: Size (in elements) of the non-contiguous (stride) dimension

        The descriptor layout (SMEMDescriptor union):
            bits[62:63] swizzlingMode: {0:none, 1:128B, 2:64B, 3:32B}
            bits[32:45] strideDimensionBaseOffset: swizzle_bytes >> 1
            bits[16:29] leadDimensionBaseOffset: (swizzle_bytes * stride_dim_size) >> 4
        """
        desc = 0
        if swizzle_bytes == 128:
            desc |= (1 << 62)
        elif swizzle_bytes == 64:
            desc |= (2 << 62)
        elif swizzle_bytes == 32:
            desc |= (3 << 62)
        desc |= ((swizzle_bytes >> 1) & 0x3FFF) << 32
        desc |= (((swizzle_bytes * stride_dim_size) >> 4) & 0x3FFF) << 16
        return desc

    def _emit_warp_group_dot_wait(self, op: IROperation):
        """Emit wgmma.wait_group.sync.aligned N."""
        pendings = 0
        p_match = re.search(r'pendings\s*=\s*(\d+)', op.raw_text)
        if p_match:
            pendings = int(p_match.group(1))
        self._emit(f'asm volatile("wgmma.wait_group.sync.aligned {pendings};");')

        # Register results - they alias the corresponding operand inputs
        # For multi-result like %result:3, register %result#0, %result#1, %result#2
        if op.results and op.operands:
            result_name = op.results[0].name
            base_name = result_name.split(':')[0]
            n_results = 1
            colon_match = re.search(r':(\d+)', result_name)
            if colon_match:
                n_results = int(colon_match.group(1))

            for i in range(min(n_results, len(op.operands))):
                src = op.operands[i]
                src_var = self._get_var(src)
                reg_name = f'{base_name}#{i}' if n_results > 1 else base_name
                self.ssa_to_var[reg_name] = src_var
                if src in self.ssa_to_type:
                    self.ssa_to_type[reg_name] = self.ssa_to_type[src]
                self.ssa_is_tensor[reg_name] = self.ssa_is_tensor.get(src, False)
                if src in self.ssa_tensor_info:
                    self.ssa_tensor_info[reg_name] = self.ssa_tensor_info[src]
                if self.ssa_is_ptr_tensor.get(src, False):
                    self.ssa_is_ptr_tensor[reg_name] = True

    def _emit_fence_async_shared(self, op: IROperation):
        """Emit fence.proxy.async.shared::cta (or ::cluster)."""
        if 'bCluster = true' in op.raw_text:
            self._emit('asm volatile("fence.proxy.async.shared::cluster;");')
        else:
            self._emit('asm volatile("fence.proxy.async.shared::cta;");')

    def _extract_barrier_ops(self, op: IROperation):
        """Extract operands from barrier ops, falling back to raw text parsing."""
        return op.operands or re.findall(r'%[\w.\-]+(?:[:#]\d+)?', op.raw_text.split(':')[0] if ':' in op.raw_text else op.raw_text)

    def _emit_init_barrier(self, op: IROperation):
        """Emit mbarrier init. Only thread 0 executes, then __syncthreads()."""
        bar_ops = self._extract_barrier_ops(op)
        if bar_ops:
            bar_var = self._get_var(bar_ops[0])
            count_m = re.search(r',\s*(\d+)', op.raw_text)
            count = int(count_m.group(1)) if count_m else 1
            self._emit(f'if (threadIdx.x == 0) {{')
            self.indent_level += 1
            self._emit(f'asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared({bar_var})), "r"({count}));')
            self.indent_level -= 1
            self._emit(f'}}')
            self._emit(f'__syncthreads();')

    def _emit_wait_barrier(self, op: IROperation):
        """Emit mbarrier wait. All threads wait."""
        bar_ops = self._extract_barrier_ops(op)
        if bar_ops:
            bar_var = self._get_var(bar_ops[0])
            phase = self._get_var(bar_ops[1]) if len(bar_ops) > 1 else '0'
            self._emit(f'{{')
            self.indent_level += 1
            self._emit(f'uint32_t bar_addr = (unsigned)__cvta_generic_to_shared({bar_var});')
            self._emit(f'asm volatile(')
            self._emit(f'    "{{\\n"')
            self._emit(f'    ".reg .pred P1;\\n"')
            self._emit(f'    "WAIT_LOOP_%=:\\n"')
            self._emit(f'    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\\n"')
            self._emit(f'    "@!P1 bra WAIT_LOOP_%=;\\n"')
            self._emit(f'    "}}\\n"')
            self._emit(f'    :: "r"(bar_addr), "r"((int){phase}));')
            self.indent_level -= 1
            self._emit(f'}}')

    def _emit_arrive_barrier(self, op: IROperation):
        """Emit mbarrier arrive. All threads arrive."""
        bar_ops = self._extract_barrier_ops(op)
        if bar_ops:
            bar_var = self._get_var(bar_ops[0])
            self._emit(f'asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"((unsigned)__cvta_generic_to_shared({bar_var})));')

    def _emit_barrier_expect(self, op: IROperation):
        """Emit mbarrier expect_tx. Only thread 0, with optional runtime predicate."""
        bar_ops = self._extract_barrier_ops(op)
        if not bar_ops:
            return
        bar_var = self._get_var(bar_ops[0])
        # Extract byte count (integer after barrier operand)
        raw_after_bar = op.raw_text.split(bar_ops[0].lstrip('%'), 1)[-1] if bar_ops else op.raw_text
        count_match = re.search(r',\s*(\d+)', raw_after_bar)
        count_var = count_match.group(1) if count_match else (self._get_var(bar_ops[1]) if len(bar_ops) > 1 else '0')
        # Extract runtime predicate (third operand after the byte count)
        pred_var = None
        if len(bar_ops) >= 3:
            pred_var = self._get_var(bar_ops[2])
        elif count_match:
            # Look for a %var after the count
            rest = raw_after_bar[count_match.end():]
            pred_m = re.search(r'(%[\w.\-]+)', rest)
            if pred_m:
                pred_var = self._get_var(pred_m.group(1))

        self._emit(f'if (threadIdx.x == 0) {{')
        self.indent_level += 1
        if pred_var:
            self._emit(f'if ({pred_var}) {{')
            self.indent_level += 1
        self._emit(f'asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared({bar_var})), "r"({count_var}));')
        if pred_var:
            self.indent_level -= 1
            self._emit(f'}}')
        self.indent_level -= 1
        self._emit(f'}}')

    def _emit_tma_copy_g2l(self, op: IROperation):
        """Emit TMA global-to-shared copy: cp.async.bulk.tensor.2d PTX.

        TTGIR: ttng.async_tma_copy_global_to_local %desc[%x, %y] %smem, %bar, %pred
        Only thread 0 executes. Runtime predicate guards the copy.
        """
        raw = op.raw_text
        operands = op.operands

        # Extract desc (first operand)
        desc_var = self._get_var(operands[0]) if operands else 'nullptr'

        # Extract coordinates from [%x, %y] in raw text
        coord_match = re.search(r'\[(%[\w.\-]+(?:[:#]\d+)?),\s*(%[\w.\-]+(?:[:#]\d+)?)\]', raw)
        coord0 = self._get_var(coord_match.group(1)) if coord_match else '0'
        coord1 = self._get_var(coord_match.group(2)) if coord_match else '0'

        # Extract: smem_dest, barrier, pred from operands after the bracket
        after_bracket = raw.split(']', 1)[-1] if ']' in raw else raw
        # Split on ':' to avoid matching type annotations, but be careful with '::' in types
        type_start = after_bracket.find(' : ')
        if type_start >= 0:
            after_bracket = after_bracket[:type_start]
        extra_ops = re.findall(r'%[\w.\-]+(?:[:#]\d+)?', after_bracket)
        smem_var = self._get_var(extra_ops[0]) if len(extra_ops) > 0 else 'nullptr'
        bar_var = self._get_var(extra_ops[1]) if len(extra_ops) > 1 else 'nullptr'
        pred_var = self._get_var(extra_ops[2]) if len(extra_ops) > 2 else None

        self._emit(f'// TMA: cp.async.bulk.tensor.2d global→shared')
        self._emit(f'if (threadIdx.x == 0) {{')
        self.indent_level += 1
        if pred_var:
            self._emit(f'if ({pred_var}) {{')
            self.indent_level += 1
        self._emit(f'asm volatile(')
        self._emit(f'    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {{%2, %3}}], [%4];\\n"')
        self._emit(f'    :: "r"((unsigned)__cvta_generic_to_shared({smem_var})),')
        self._emit(f'       "l"((uint64_t)&{desc_var}),')
        self._emit(f'       "r"({coord1}), "r"({coord0}),')
        self._emit(f'       "r"((unsigned)__cvta_generic_to_shared({bar_var}))')
        self._emit(f');')
        if pred_var:
            self.indent_level -= 1
            self._emit(f'}}')
        self.indent_level -= 1
        self._emit(f'}}')

    def _emit_tma_copy_l2g(self, op: IROperation):
        """Emit TMA shared-to-global copy: cp.async.bulk.tensor.2d PTX.

        TTGIR: ttng.async_tma_copy_local_to_global %desc[%x, %y] %smem
        """
        raw = op.raw_text
        operands = op.operands
        desc_var = self._get_var(operands[0]) if operands else 'nullptr'

        coord_match = re.search(r'\[(%[\w.\-]+(?:[:#]\d+)?),\s*(%[\w.\-]+(?:[:#]\d+)?)\]', raw)
        coord0 = self._get_var(coord_match.group(1)) if coord_match else '0'
        coord1 = self._get_var(coord_match.group(2)) if coord_match else '0'

        after_bracket = raw.split(']', 1)[-1] if ']' in raw else raw
        extra_ops = re.findall(r'%[\w.\-]+(?:[:#]\d+)?', after_bracket.split(':')[0] if ':' in after_bracket else after_bracket)
        smem_var = self._get_var(extra_ops[0]) if extra_ops else 'nullptr'

        self._emit(f'// TMA: cp.async.bulk.tensor.2d shared→global + commit')
        self._emit(f'if (threadIdx.x == 0) {{')
        self.indent_level += 1
        self._emit(f'asm volatile(')
        self._emit(f'    "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {{%1, %2}}], [%3];\\n"')
        self._emit(f'    :: "l"((uint64_t)&{desc_var}),')
        self._emit(f'       "r"({coord1}), "r"({coord0}),')
        self._emit(f'       "r"((unsigned)__cvta_generic_to_shared({smem_var}))')
        self._emit(f');')
        self._emit(f'asm volatile("cp.async.bulk.commit_group;");')
        self.indent_level -= 1
        self._emit(f'}}')

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _extract_tensor_type(self, type_str: str) -> Optional[TensorType]:
        """Extract TensorType from a type string like 'tensor<1024xf32, #blocked>'."""
        if not type_str:
            return None
        # Find tensor type, handling nested <> (e.g., tensor<128x!tt.ptr<f16>, #blocked>)
        start = type_str.find('tensor<')
        if start < 0:
            return None
        # Find matching >
        depth = 0
        end = start + 7  # len('tensor<')
        for i in range(start + 7, len(type_str)):
            if type_str[i] == '<':
                depth += 1
            elif type_str[i] == '>':
                if depth == 0:
                    end = i
                    break
                depth -= 1
        inner = type_str[start + 7:end]
        if not inner:
            return None
        # Split on last comma to separate shape from layout
        # Use simple comma split respecting nesting
        parts = _split_args(inner)
        shape_part = parts[0].strip()
        layout = None
        if len(parts) > 1:
            layout_ref = parts[-1].strip()
            if layout_ref in self.module.layout_aliases:
                layout = self.module.layout_aliases[layout_ref]
            elif '#ttg.slice' in layout_ref:
                # Inline slice layout: #ttg.slice<{dim = N, parent = #xxx}>
                dim_m = re.search(r'dim\s*=\s*(\d+)', layout_ref)
                parent_m = re.search(r'parent\s*=\s*(#\w+)', layout_ref)
                if dim_m and parent_m:
                    parent_layout = self.module.layout_aliases.get(parent_m.group(1))
                    layout = SliceLayout(parent=parent_layout, dim=int(dim_m.group(1)))

        # Parse shape - handle cases like "1024x!tt.ptr<f32>" or "1024xf32"
        # Split on 'x' but not inside < >
        element_type = shape_part
        shape = []
        # Try extracting numeric dimensions from the front
        remaining = shape_part
        while remaining:
            m = re.match(r'(\d+)x(.*)', remaining)
            if m:
                shape.append(int(m.group(1)))
                remaining = m.group(2)
            else:
                element_type = remaining.strip()
                break

        # Keep ptr types as-is (mlir_type_to_cuda handles them)
        return TensorType(shape=shape, element_type=element_type, layout=layout)

    def _get_elems_for_tensor(self, tt: TensorType) -> int:
        """Get number of per-thread elements for a tensor type."""
        if isinstance(tt.layout, BlockedLayout):
            return tt.layout.elems_per_thread(tt.shape)
        elif isinstance(tt.layout, SliceLayout):
            # Slice removes one dim from parent layout
            parent = tt.layout.parent
            if isinstance(parent, BlockedLayout):
                # Reconstruct parent shape by inserting 1 at sliced dim
                parent_shape = list(tt.shape)
                parent_shape.insert(tt.layout.dim, 1)
                parent_elems = parent.elems_per_thread(parent_shape)
                return max(parent_elems, 1)
            # Fallback for slice of non-blocked
            total = 1
            for s in tt.shape:
                total *= s
            num_threads = self.module.num_warps * self.module.threads_per_warp
            return max(total // num_threads, 1)
        elif isinstance(tt.layout, MMALayout):
            # For MMA layout, compute based on instruction shape
            total = 1
            for s in tt.shape:
                total *= s
            num_threads = self.module.num_warps * self.module.threads_per_warp
            return max(total // num_threads, 1)
        # Default: total elements / num_threads
        total = 1
        for s in tt.shape:
            total *= s
        num_threads = self.module.num_warps * self.module.threads_per_warp
        return max(total // num_threads, 1)

    def _get_num_elems(self, ssa_name: str) -> int:
        """Get number of per-thread elements for a registered SSA value."""
        if ssa_name in self.ssa_tensor_info:
            shape, layout = self.ssa_tensor_info[ssa_name]
            if isinstance(layout, BlockedLayout):
                return layout.elems_per_thread(shape)
            total = 1
            for s in shape:
                total *= s
            num_threads = self.module.num_warps * self.module.threads_per_warp
            return max(total // num_threads, 1)
        return 1

    def _extract_result_type(self, op: IROperation) -> str:
        """Extract the result element type from an op's type annotation."""
        raw = op.raw_text
        # For conversion ops like "arith.extui : i32 to i64", result type is after "to"
        m_to = re.search(r'\bto\s+(tensor<[^>]+>|\S+)\s*$', raw)
        if m_to:
            t = m_to.group(1).strip()
            if t.startswith('tensor<'):
                return self._get_scalar_type(t)
            if t in MLIR_TO_CUDA_TYPE:
                return t
        # Look for the result type after ->
        m = re.search(r'->\s*(\S+)', raw)
        if m:
            t = m.group(1).strip()
            if t.startswith('tensor<'):
                return self._get_scalar_type(t)
            return t
        # Look for : type at the end
        m = re.search(r':\s*(\S+)\s*$', raw)
        if m:
            t = m.group(1).strip()
            if t in MLIR_TO_CUDA_TYPE:
                return t
        return 'f32'


# ---------------------------------------------------------------------------
# Top-level CUDA Emitter
# ---------------------------------------------------------------------------

class CUDAEmitter:
    """Top-level orchestrator: takes an MLIR module and produces CUDA source code."""

    def __init__(self, capability: int = 80, num_warps: int = 4, num_ctas: int = 1):
        self.capability = capability
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.kernel_name = ""
        self.shared_mem_size = 0

    def emit(self, mod) -> str:
        """Main entry point: takes MLIR module, returns CUDA source code string."""
        # Get text representation
        mlir_text = str(mod)

        # Parse MLIR text
        parser = MLIRTextParser()
        ir_module = parser.parse_module(mlir_text)

        # Override with constructor params
        ir_module.num_warps = self.num_warps
        ir_module.num_ctas = self.num_ctas

        # Generate CUDA code
        codegen = CUDACodeGen(ir_module, self.capability)
        cuda_src = codegen.generate()

        # Extract metadata
        self.kernel_name = codegen.kernel_name
        # shared_mem_size: use the larger of module attribute and emitter's own tracking
        self.shared_mem_size = max(ir_module.shared_size, codegen.shared_mem_offset)

        return cuda_src
