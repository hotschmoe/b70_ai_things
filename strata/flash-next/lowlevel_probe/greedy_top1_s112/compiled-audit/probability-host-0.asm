
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/greedy-top1-s112-oracle-prep-v2-built-audit/verify_kernels.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000002cb0 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_>:
    2cb0:	55                   	push   %rbp
    2cb1:	41 57                	push   %r15
    2cb3:	41 56                	push   %r14
    2cb5:	41 55                	push   %r13
    2cb7:	41 54                	push   %r12
    2cb9:	53                   	push   %rbx
    2cba:	48 81 ec c8 01 00 00 	sub    $0x1c8,%rsp
    2cc1:	85 f6                	test   %esi,%esi
    2cc3:	0f 8e ec 02 00 00    	jle    2fb5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x305>
    2cc9:	4d 89 ce             	mov    %r9,%r14
    2ccc:	4d 89 c7             	mov    %r8,%r15
    2ccf:	49 89 cc             	mov    %rcx,%r12
    2cd2:	41 89 d5             	mov    %edx,%r13d
    2cd5:	89 f5                	mov    %esi,%ebp
    2cd7:	48 89 fb             	mov    %rdi,%rbx
    2cda:	48 8b 84 24 00 02 00 00 	mov    0x200(%rsp),%rax
    2ce2:	48 85 c0             	test   %rax,%rax
    2ce5:	75 6e                	jne    2d55 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0xa5>
    2ce7:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 2cee <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3e>	2cea: R_X86_64_PC32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m-0x4
    2cee:	84 c0                	test   %al,%al
    2cf0:	0f 84 1c 03 00 00    	je     3012 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x362>
    2cf6:	e8 00 00 00 00       	call   2cfb <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x4b>	2cf7: R_X86_64_PLT32	_ZTHN4dpct7dev_mgr10_dev_stackE-0x4
    2cfb:	64 48 8b 04 25 00 00 00 00 	mov    %fs:0x0,%rax	2d00: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x30
    2d04:	64 48 3b 04 25 00 00 00 00 	cmp    %fs:0x0,%rax	2d09: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x10
    2d0d:	74 31                	je     2d40 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x90>
    2d0f:	e8 00 00 00 00       	call   2d14 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x64>	2d10: R_X86_64_PLT32	_ZTHN4dpct7dev_mgr10_dev_stackE-0x4
    2d14:	64 48 8b 04 25 00 00 00 00 	mov    %fs:0x0,%rax	2d19: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x30
    2d1d:	64 48 3b 04 25 00 00 00 00 	cmp    %fs:0x0,%rax	2d22: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x38
    2d26:	75 12                	jne    2d3a <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x8a>
    2d28:	64 48 8b 0c 25 00 00 00 00 	mov    %fs:0x0,%rcx	2d2d: R_X86_64_TPOFF32	_ZN4dpct7dev_mgr10_dev_stackE+0x48
    2d31:	b8 00 02 00 00       	mov    $0x200,%eax
    2d36:	48 03 41 f8          	add    -0x8(%rcx),%rax
    2d3a:	48 83 c0 fc          	add    $0xfffffffffffffffc,%rax
    2d3e:	eb 05                	jmp    2d45 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x95>
    2d40:	b8 00 00 00 00       	mov    $0x0,%eax	2d41: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m+0x40
    2d45:	8b 30                	mov    (%rax),%esi
    2d47:	bf 00 00 00 00       	mov    $0x0,%edi	2d48: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    2d4c:	e8 00 00 00 00       	call   2d51 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0xa1>	2d4d: R_X86_64_PLT32	_ZNK4dpct7dev_mgr10get_deviceEj-0x4
    2d51:	48 8b 40 08          	mov    0x8(%rax),%rax
    2d55:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    2d5a:	48 89 9c 24 c8 00 00 00 	mov    %rbx,0xc8(%rsp)
    2d62:	44 89 ac 24 d0 00 00 00 	mov    %r13d,0xd0(%rsp)
    2d6a:	4c 89 a4 24 d8 00 00 00 	mov    %r12,0xd8(%rsp)
    2d72:	4c 89 bc 24 e0 00 00 00 	mov    %r15,0xe0(%rsp)
    2d7a:	4c 89 b4 24 e8 00 00 00 	mov    %r14,0xe8(%rsp)
    2d82:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 2d89 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0xd9>	2d85: R_X86_64_PC32	.rodata+0x4ffc
    2d89:	48 89 84 24 00 01 00 00 	mov    %rax,0x100(%rsp)
    2d91:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 2d98 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0xe8>	2d94: R_X86_64_PC32	.rodata+0x4fec
    2d98:	0f 29 84 24 f0 00 00 00 	movaps %xmm0,0xf0(%rsp)
    2da0:	48 8d 5c 24 40       	lea    0x40(%rsp),%rbx
    2da5:	48 8d b4 24 f0 00 00 00 	lea    0xf0(%rsp),%rsi
    2dad:	48 89 df             	mov    %rbx,%rdi
    2db0:	e8 00 00 00 00       	call   2db5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x105>	2db1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    2db5:	48 89 df             	mov    %rbx,%rdi
    2db8:	e8 00 00 00 00       	call   2dbd <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x10d>	2db9: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    2dbd:	48 89 c3             	mov    %rax,%rbx
    2dc0:	89 e8                	mov    %ebp,%eax
    2dc2:	48 c7 84 24 80 00 00 00 01 00 00 00 	movq   $0x1,0x80(%rsp)
    2dce:	48 89 84 24 88 00 00 00 	mov    %rax,0x88(%rsp)
    2dd6:	48 c7 84 24 90 00 00 00 00 04 00 00 	movq   $0x400,0x90(%rsp)
    2de2:	48 c7 84 24 98 00 00 00 01 00 00 00 	movq   $0x1,0x98(%rsp)
    2dee:	48 c7 84 24 a0 00 00 00 01 00 00 00 	movq   $0x1,0xa0(%rsp)
    2dfa:	48 c7 84 24 a8 00 00 00 80 00 00 00 	movq   $0x80,0xa8(%rsp)
    2e06:	0f 57 c0             	xorps  %xmm0,%xmm0
    2e09:	0f 11 84 24 b0 00 00 00 	movups %xmm0,0xb0(%rsp)
    2e11:	48 c7 84 24 c0 00 00 00 00 00 00 00 	movq   $0x0,0xc0(%rsp)
    2e1d:	4c 8d b4 24 80 00 00 00 	lea    0x80(%rsp),%r14
    2e25:	4c 89 f7             	mov    %r14,%rdi
    2e28:	e8 00 00 00 00       	call   2e2d <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x17d>	2e29: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
    2e2d:	48 8d 84 24 98 00 00 00 	lea    0x98(%rsp),%rax
    2e35:	48 8d 8c 24 b0 00 00 00 	lea    0xb0(%rsp),%rcx
    2e3d:	4c 89 74 24 60       	mov    %r14,0x60(%rsp)
    2e42:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
    2e47:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
    2e4c:	48 c7 44 24 78 03 00 00 00 	movq   $0x3,0x78(%rsp)
    2e55:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
    2e5a:	48 89 de             	mov    %rbx,%rsi
    2e5d:	e8 00 00 00 00       	call   2e62 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x1b2>	2e5e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
    2e62:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 2e69 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x1b9>	2e65: R_X86_64_PC32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    2e69:	84 c0                	test   %al,%al
    2e6b:	0f 84 6f 01 00 00    	je     2fe0 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x330>
    2e71:	48 8b 1d 00 00 00 00 	mov    0x0(%rip),%rbx        # 2e78 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x1c8>	2e74: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    2e78:	c6 84 24 38 01 00 00 00 	movb   $0x0,0x138(%rsp)
    2e80:	c6 84 24 42 01 00 00 00 	movb   $0x0,0x142(%rsp)
    2e88:	c6 84 24 44 01 00 00 01 	movb   $0x1,0x144(%rsp)
    2e90:	c6 84 24 4c 01 00 00 00 	movb   $0x0,0x14c(%rsp)
    2e98:	c6 84 24 54 01 00 00 00 	movb   $0x0,0x154(%rsp)
    2ea0:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
    2ea8:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
    2eb0:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
    2eb8:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
    2ec0:	c6 84 24 80 01 00 00 00 	movb   $0x0,0x180(%rsp)
    2ec8:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
    2ed0:	c6 84 24 b8 01 00 00 00 	movb   $0x0,0x1b8(%rsp)
    2ed8:	c6 84 24 c0 01 00 00 00 	movb   $0x0,0x1c0(%rsp)
    2ee0:	48 c7 44 24 50 00 00 00 00 	movq   $0x0,0x50(%rsp)	2ee5: R_X86_64_32S	.rodata+0x22c0
    2ee9:	48 8d 84 24 c8 00 00 00 	lea    0xc8(%rsp),%rax
    2ef1:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    2ef6:	0f 57 c0             	xorps  %xmm0,%xmm0
    2ef9:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
    2f01:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
    2f06:	e8 00 00 00 00       	call   2f0b <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x25b>	2f07: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
    2f0b:	31 c9                	xor    %ecx,%ecx
    2f0d:	80 7c 24 30 00       	cmpb   $0x0,0x30(%rsp)
    2f12:	0f 94 c1             	sete   %cl
    2f15:	0f 28 84 24 10 01 00 00 	movaps 0x110(%rsp),%xmm0
    2f1d:	0f 11 04 24          	movups %xmm0,(%rsp)
    2f21:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
    2f25:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    2f2a:	48 8d bc 24 20 01 00 00 	lea    0x120(%rsp),%rdi
    2f32:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
    2f37:	48 8d 4c 24 50       	lea    0x50(%rsp),%rcx
    2f3c:	4c 8d 8c 24 30 01 00 00 	lea    0x130(%rsp),%r9
    2f44:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    2f49:	49 89 d8             	mov    %rbx,%r8
    2f4c:	e8 00 00 00 00       	call   2f51 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x2a1>	2f4d: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
    2f51:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
    2f56:	e8 00 00 00 00       	call   2f5b <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x2ab>	2f57: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    2f5b:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    2f60:	e8 00 00 00 00       	call   2f65 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x2b5>	2f61: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    2f65:	48 8b 9c 24 28 01 00 00 	mov    0x128(%rsp),%rbx
    2f6d:	48 85 db             	test   %rbx,%rbx
    2f70:	74 43                	je     2fb5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x305>
    2f72:	48 8b 43 08          	mov    0x8(%rbx),%rax
    2f76:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
    2f80:	48 39 c8             	cmp    %rcx,%rax
    2f83:	75 1c                	jne    2fa1 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x2f1>
    2f85:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
    2f8d:	48 8b 03             	mov    (%rbx),%rax
    2f90:	48 89 df             	mov    %rbx,%rdi
    2f93:	ff 50 10             	call   *0x10(%rax)
    2f96:	48 8b 03             	mov    (%rbx),%rax
    2f99:	48 89 df             	mov    %rbx,%rdi
    2f9c:	ff 50 18             	call   *0x18(%rax)
    2f9f:	eb 14                	jmp    2fb5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x305>
    2fa1:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 2fa8 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x2f8>	2fa3: R_X86_64_PC32	__libc_single_threaded-0x5
    2fa8:	74 1d                	je     2fc7 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x317>
    2faa:	8d 48 ff             	lea    -0x1(%rax),%ecx
    2fad:	89 4b 08             	mov    %ecx,0x8(%rbx)
    2fb0:	83 f8 01             	cmp    $0x1,%eax
    2fb3:	74 21                	je     2fd6 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x326>
    2fb5:	48 81 c4 c8 01 00 00 	add    $0x1c8,%rsp
    2fbc:	5b                   	pop    %rbx
    2fbd:	41 5c                	pop    %r12
    2fbf:	41 5d                	pop    %r13
    2fc1:	41 5e                	pop    %r14
    2fc3:	41 5f                	pop    %r15
    2fc5:	5d                   	pop    %rbp
    2fc6:	c3                   	ret
    2fc7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
    2fcc:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
    2fd1:	83 f8 01             	cmp    $0x1,%eax
    2fd4:	75 df                	jne    2fb5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x305>
    2fd6:	48 89 df             	mov    %rbx,%rdi
    2fd9:	e8 00 00 00 00       	call   2fde <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x32e>	2fda: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
    2fde:	eb d5                	jmp    2fb5 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x305>
    2fe0:	bf 00 00 00 00       	mov    $0x0,%edi	2fe1: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info
    2fe5:	e8 00 00 00 00       	call   2fea <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x33a>	2fe6: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    2fea:	85 c0                	test   %eax,%eax
    2fec:	0f 84 7f fe ff ff    	je     2e71 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x1c1>
    2ff2:	bf 00 00 00 00       	mov    $0x0,%edi	2ff3: R_X86_64_32	_ZN4sycl3_V16detail27compile_time_kernel_info_v121CompileTimeKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEE
    2ff7:	e8 00 00 00 00       	call   2ffc <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x34c>	2ff8: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
    2ffc:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 3003 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x353>	2fff: R_X86_64_PC32	_ZZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info-0x4
    3003:	bf 00 00 00 00       	mov    $0x0,%edi	3004: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info
    3008:	e8 00 00 00 00       	call   300d <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x35d>	3009: R_X86_64_PLT32	__cxa_guard_release-0x4
    300d:	e9 5f fe ff ff       	jmp    2e71 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x1c1>
    3012:	bf 00 00 00 00       	mov    $0x0,%edi	3013: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    3017:	e8 00 00 00 00       	call   301c <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x36c>	3018: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    301c:	85 c0                	test   %eax,%eax
    301e:	0f 84 d2 fc ff ff    	je     2cf6 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x46>
    3024:	bf 00 00 00 00       	mov    $0x0,%edi	3025: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    3029:	e8 00 00 00 00       	call   302e <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x37e>	302a: R_X86_64_PLT32	_ZN4dpct7dev_mgrC2Ev-0x4
    302e:	bf 00 00 00 00       	mov    $0x0,%edi	302f: R_X86_64_32	_ZN4dpct7dev_mgrD2Ev
    3033:	be 00 00 00 00       	mov    $0x0,%esi	3034: R_X86_64_32	_ZZN4dpct7dev_mgr8instanceEvE3d_m
    3038:	ba 00 00 00 00       	mov    $0x0,%edx	3039: R_X86_64_32	__dso_handle
    303d:	e8 00 00 00 00       	call   3042 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x392>	303e: R_X86_64_PLT32	__cxa_atexit-0x4
    3042:	bf 00 00 00 00       	mov    $0x0,%edi	3043: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    3047:	e8 00 00 00 00       	call   304c <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x39c>	3048: R_X86_64_PLT32	__cxa_guard_release-0x4
    304c:	e9 a5 fc ff ff       	jmp    2cf6 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x46>
    3051:	48 89 c3             	mov    %rax,%rbx
    3054:	bf 00 00 00 00       	mov    $0x0,%edi	3055: R_X86_64_32	_ZGVZN4dpct7dev_mgr8instanceEvE3d_m
    3059:	e8 00 00 00 00       	call   305e <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3ae>	305a: R_X86_64_PLT32	__cxa_guard_abort-0x4
    305e:	48 89 df             	mov    %rbx,%rdi
    3061:	e8 00 00 00 00       	call   3066 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3b6>	3062: R_X86_64_PLT32	_Unwind_Resume-0x4
    3066:	48 89 c3             	mov    %rax,%rbx
    3069:	bf 00 00 00 00       	mov    $0x0,%edi	306a: R_X86_64_32	_ZGVZN4sycl3_V16detail19getDeviceKernelInfoI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEERNS1_16DeviceKernelInfoEvE4Info
    306e:	e8 00 00 00 00       	call   3073 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3c3>	306f: R_X86_64_PLT32	__cxa_guard_abort-0x4
    3073:	eb 03                	jmp    3078 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3c8>
    3075:	48 89 c3             	mov    %rax,%rbx
    3078:	48 8d 7c 24 28       	lea    0x28(%rsp),%rdi
    307d:	e8 00 00 00 00       	call   3082 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3d2>	307e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    3082:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    3087:	e8 00 00 00 00       	call   308c <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3dc>	3088: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    308c:	48 89 df             	mov    %rbx,%rdi
    308f:	e8 00 00 00 00       	call   3094 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3e4>	3090: R_X86_64_PLT32	_Unwind_Resume-0x4
    3094:	48 89 c3             	mov    %rax,%rbx
    3097:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    309c:	e8 00 00 00 00       	call   30a1 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3f1>	309d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
    30a1:	48 89 df             	mov    %rbx,%rdi
    30a4:	e8 00 00 00 00       	call   30a9 <_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_+0x3f9>	30a5: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZN6strata8spin_maxERKN4sycl3_V15queueE:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

Disassembly of section .text._ZN4dpct7dev_mgrC2Ev:

Disassembly of section .text._ZN4dpct7dev_mgrD2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4dpct10device_extEESaIS3_EE9push_backEOS3_:

Disassembly of section .text._ZSt11make_sharedIN4dpct10device_extEJRN4sycl3_V16deviceEEESt10shared_ptrIT_EDpOT0_:

Disassembly of section .text._ZNSt12__shared_ptrIN4dpct10device_extELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4dpct10device_extEESaIS3_EED2Ev:

Disassembly of section .text._ZNSt17_Function_handlerIFiRKN4sycl3_V16deviceEEPS5_E9_M_invokeERKSt9_Any_dataS4_:

Disassembly of section .text._ZNSt17_Function_handlerIFiRKN4sycl3_V16deviceEEPS5_E10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4dpct10device_extESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZN4sycl3_V16detail16PropertyListBaseD2Ev:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EED2Ev:

Disassembly of section .text._ZN4sycl3_V17contextD2Ev:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EE9push_backEOS4_:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V15queueELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZN4sycl3_V114exception_listD2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZNK4dpct7dev_mgr10get_deviceEj:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev:

Disassembly of section .text._ZNSt13unordered_mapIN4sycl3_V16deviceEjSt4hashIS2_ESt8equal_toIS2_ESaISt4pairIKS2_jEEED2Ev:

Disassembly of section .text._ZZN6strata16intel_gpu_driverB5cxx11EvENKUlvE_clB5cxx11Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathD2Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx1118directory_iteratorD2Ev:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathC2IA7_cS1_EERKT_NS1_6formatE:

Disassembly of section .text._ZNSt10filesystem7__cxx114pathC2ERKS1_:

Disassembly of section .text._ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4findEPKcm:

Disassembly of section .text._ZNSt6thread24_M_thread_deps_never_runEv:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25fetch_blobs_kernel_14b6f5EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25rebase_ptrs_kernel_436f27EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels35add_streams_broadcast_kernel_7de8f5EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24ident_hits_kernel_287489EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31copy_row_to_first_kernel_993785EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24mtp_select_kernel_4611feEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25force_token_kernel_a9fa72EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_b3ae5aENS0_3vecIjLi4EEEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_f256acEjEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25gather_rows_kernel_c5b171EhEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels21map_ids_kernel_43a113EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26row_top_prob_kernel_6adbe2EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25argmax_rows_kernel_d5bcf6EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32row_top_prob_split_kernel_9495b0EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels24window_ids_kernel_581536EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels25dense_steps_kernel_fd7245EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31gdn_conv_l2_multi_kernel_b3f737EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29gdn_conv_commit_kernel_dc1d7fEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi1EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi2EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi3EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi4EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi5EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi6EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_ab_multi_kernel_34dcceE18dpct_kernel_scalarILi8EES7_ILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30gdn_state_replay_kernel_e4d4c4E18dpct_kernel_scalarILi8EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels28gdn_step_split_kernel_83c676E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_out_norm_kernel_a650ebE18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26gdn_out_norm_kernel_f6dfabE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29gdn_step_commit_kernel_5d9da7EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi1EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi0EES7_ILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi1EES7_ILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels33gdn_step_norm_multi_kernel_42ae26E18dpct_kernel_scalarILi0EES8_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels27resident_plan_kernel_db6b7aEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels23resident_plan_ep_kernelEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29wait_flag_ge_or_kernel_2b2de3EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels29copy_i32_unless_kernel_d9d761EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26copy_or_zero_kernel_5ef081EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26wait_flag_ge_kernel_d7debfEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels34embedding_gather_dev_kernel_600016EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels31broadcast_streams_kernel_aa4f6fEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels26copy_indexed_kernel_b12e23EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:
