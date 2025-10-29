; ModuleID = './3rdparty/triton/third_party/amd/backend/lib/libdevice_extra.ll'
source_filename = "llvm-link"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
; compile this with `hipcc -S -emit-llvm __gen_smid.hip -o -`
;  __global__ void __extra_smid(int *ptr) {   ptr[0] = __smid(); }
define linkonce hidden i32 @__extra_smid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 2884)
  %3 = tail call i32 @llvm.amdgcn.s.getreg(i32 6164)
  %4 = tail call i32 @llvm.amdgcn.s.getreg(i32 6660)
  %5 = shl i32 %3, 6
  %6 = shl i32 %2, 4
  %7 = or i32 %5, %6
  %8 = or i32 %7, %4
  ret i32 %8
}

define linkonce hidden i32 @__extra_seid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 2884)
  ret i32 %2
}

define linkonce hidden i32 @__extra_xccid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 6164)
  ret i32 %2
}

define linkonce hidden i32 @__extra_cuid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 6660)
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden i64 @__extra_clock() local_unnamed_addr #3 {
  %2 = tail call noundef i64 @llvm.amdgcn.s.memtime()
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden i64 @__extra_wallclock() local_unnamed_addr #3 {
  %2 = tail call noundef i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden void @__extra_fence_system() local_unnamed_addr {
  fence syncscope("agent") release
  ret void
}

define linkonce hidden i32 @__load_acquire_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("workgroup-one-as") acquire, align 4
  ret i32 %2
}


define linkonce hidden i32 @__load_acquire_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("agent-one-as") acquire, align 4
  ret i32 %2
}


define linkonce hidden i32 @__load_acquire_system_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0  acquire, align 4
  ret i32 %2
}


define linkonce hidden i32 @__load_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 4
  ret i32 %2
}


define linkonce hidden i32 @__load_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 4
  ret i32 %2
}


define linkonce hidden i32 @__load_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i32, ptr addrspace(1) %0  monotonic, align 4
  ret i32 %2
}


define linkonce hidden i64 @__load_acquire_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("workgroup-one-as") acquire, align 8
  ret i64 %2
}


define linkonce hidden i64 @__load_acquire_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("agent-one-as") acquire, align 8
  ret i64 %2
}


define linkonce hidden i64 @__load_acquire_system_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0  acquire, align 8
  ret i64 %2
}


define linkonce hidden i64 @__load_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 8
  ret i64 %2
}


define linkonce hidden i64 @__load_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 8
  ret i64 %2
}


define linkonce hidden i64 @__load_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0) local_unnamed_addr {
  %2 = load atomic i64, ptr addrspace(1) %0  monotonic, align 8
  ret i64 %2
}


define linkonce hidden i32 @__store_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 4
  ret i32 %1
}


define linkonce hidden i32 @__store_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 4
  ret i32 %1
}


define linkonce hidden i32 @__store_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0  monotonic, align 4
  ret i32 %1
}


define linkonce hidden i32 @__store_release_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("workgroup-one-as") release, align 4
  ret i32 %1
}


define linkonce hidden i32 @__store_release_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("agent-one-as") release, align 4
  ret i32 %1
}


define linkonce hidden i32 @__store_release_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  store atomic i32 %1, ptr addrspace(1) %0  release, align 4
  ret i32 %1
}


define linkonce hidden i64 @__store_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 8
  ret i64 %1
}


define linkonce hidden i64 @__store_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 8
  ret i64 %1
}


define linkonce hidden i64 @__store_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0  monotonic, align 8
  ret i64 %1
}


define linkonce hidden i64 @__store_release_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0 syncscope("workgroup-one-as") release, align 8
  ret i64 %1
}


define linkonce hidden i64 @__store_release_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0 syncscope("agent-one-as") release, align 8
  ret i64 %1
}


define linkonce hidden i64 @__store_release_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  store atomic i64 %1, ptr addrspace(1) %0  release, align 8
  ret i64 %1
}


define linkonce hidden i32 @__atomic_add_acquire_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("workgroup-one-as") acquire, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_acquire_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("agent-one-as") acquire, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_acquire_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1  acquire, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("workgroup-one-as") monotonic, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("agent-one-as") monotonic, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1  monotonic, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_release_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("workgroup-one-as") release, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_release_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("agent-one-as") release, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_release_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1  release, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_acq_rel_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("workgroup-one-as") acq_rel, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_acq_rel_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("agent-one-as") acq_rel, align 4
  ret i32 %3
}


define linkonce hidden i32 @__atomic_add_acq_rel_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1  acq_rel, align 4
  ret i32 %3
}


define linkonce hidden i64 @__atomic_add_acquire_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("workgroup-one-as") acquire, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_acquire_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("agent-one-as") acquire, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_acquire_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1  acquire, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("workgroup-one-as") monotonic, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("agent-one-as") monotonic, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1  monotonic, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_release_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("workgroup-one-as") release, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_release_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("agent-one-as") release, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_release_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1  release, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_acq_rel_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("workgroup-one-as") acq_rel, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_acq_rel_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1 syncscope("agent-one-as") acq_rel, align 8
  ret i64 %3
}


define linkonce hidden i64 @__atomic_add_acq_rel_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1) local_unnamed_addr {
  %3 = atomicrmw add ptr addrspace(1) %0, i64 %1  acq_rel, align 8
  ret i64 %3
}


define linkonce hidden i32 @__atom_cas_acquire_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("workgroup-one-as") acquire monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_acquire_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("agent-one-as") acquire monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_acquire_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2  acquire monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_monotonic_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("workgroup-one-as") monotonic monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_monotonic_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("agent-one-as") monotonic monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_monotonic_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2  monotonic monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_release_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("workgroup-one-as") release monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_release_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("agent-one-as") release monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_release_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2  release monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_acq_rel_monotonic_workgroup_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("workgroup-one-as") acq_rel monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_acq_rel_monotonic_agent_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2 syncscope("agent-one-as") acq_rel monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i32 @__atom_cas_acq_rel_monotonic_system_i32(ptr addrspace(1) inreg readonly captures(none) %0, i32 %1, i32 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i32 %1, i32 %2  acq_rel monotonic, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  ret i32 %11
}


define linkonce hidden i64 @__atom_cas_acquire_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("workgroup-one-as") acquire monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_acquire_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("agent-one-as") acquire monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_acquire_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2  acquire monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_monotonic_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("workgroup-one-as") monotonic monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_monotonic_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("agent-one-as") monotonic monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_monotonic_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2  monotonic monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_release_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("workgroup-one-as") release monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_release_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("agent-one-as") release monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_release_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2  release monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_acq_rel_monotonic_workgroup_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("workgroup-one-as") acq_rel monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_acq_rel_monotonic_agent_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2 syncscope("agent-one-as") acq_rel monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden i64 @__atom_cas_acq_rel_monotonic_system_i64(ptr addrspace(1) inreg readonly captures(none) %0, i64 %1, i64 %2) local_unnamed_addr {
  ; %1 => value
  ; %2 => target value
  %10 = cmpxchg ptr addrspace(1) %0, i64 %1, i64 %2  acq_rel monotonic, align 8
  %11 = extractvalue { i64, i1 } %10, 0
  ret i64 %11
}


define linkonce hidden void @__extra_fence_acquire_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") acquire
  ret void
}

define linkonce hidden void @__extra_fence_acquire_agent() local_unnamed_addr {
  fence syncscope("agent") acquire
  ret void
}

define linkonce hidden void @__extra_fence_acquire_system() local_unnamed_addr {
  fence syncscope("system") acquire
  ret void
}

define linkonce hidden void @__extra_fence_release_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") release
  ret void
}

define linkonce hidden void @__extra_fence_release_agent() local_unnamed_addr {
  fence syncscope("agent") release
  ret void
}

define linkonce hidden void @__extra_fence_release_system() local_unnamed_addr {
  fence syncscope("system") release
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_agent() local_unnamed_addr {
  fence syncscope("agent") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_system() local_unnamed_addr {
  fence syncscope("system") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") seq_cst
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_agent() local_unnamed_addr {
  fence syncscope("agent") seq_cst
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_system() local_unnamed_addr {
  fence syncscope("system") seq_cst
  ret void
}

attributes #3 = {
  nounwind
  "target-cpu"="gfx942"
  "uniform-work-group-size"="true"
  "target-features"="+gfx940-insts"
}
