
/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

00000000000856d0 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_>:
   856d0:	41 57                	push   %r15
   856d2:	41 56                	push   %r14
   856d4:	41 55                	push   %r13
   856d6:	41 54                	push   %r12
   856d8:	53                   	push   %rbx
   856d9:	48 81 ec 80 01 00 00 	sub    $0x180,%rsp
   856e0:	49 89 d7             	mov    %rdx,%r15
   856e3:	48 89 f3             	mov    %rsi,%rbx
   856e6:	49 89 fe             	mov    %rdi,%r14
   856e9:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 856f0 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x20>	856ec: R_X86_64_PC32	.rodata+0x786ac
   856f0:	48 89 84 24 80 00 00 00 	mov    %rax,0x80(%rsp)
   856f8:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 856ff <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x2f>	856fb: R_X86_64_PC32	.rodata+0x7869c
   856ff:	0f 29 44 24 70       	movaps %xmm0,0x70(%rsp)
   85704:	4c 8d 64 24 30       	lea    0x30(%rsp),%r12
   85709:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
   8570e:	4c 89 e7             	mov    %r12,%rdi
   85711:	e8 00 00 00 00       	call   85716 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x46>	85712: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   85716:	4c 89 e7             	mov    %r12,%rdi
   85719:	e8 00 00 00 00       	call   8571e <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x4e>	8571a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   8571e:	49 89 c4             	mov    %rax,%r12
   85721:	48 8d 84 24 b0 01 00 00 	lea    0x1b0(%rsp),%rax
   85729:	48 8b 48 40          	mov    0x40(%rax),%rcx
   8572d:	48 89 8c 24 e0 00 00 00 	mov    %rcx,0xe0(%rsp)
   85735:	0f 10 00             	movups (%rax),%xmm0
   85738:	0f 10 48 10          	movups 0x10(%rax),%xmm1
   8573c:	0f 10 50 20          	movups 0x20(%rax),%xmm2
   85740:	0f 10 58 30          	movups 0x30(%rax),%xmm3
   85744:	0f 29 9c 24 d0 00 00 00 	movaps %xmm3,0xd0(%rsp)
   8574c:	0f 29 94 24 c0 00 00 00 	movaps %xmm2,0xc0(%rsp)
   85754:	0f 29 8c 24 b0 00 00 00 	movaps %xmm1,0xb0(%rsp)
   8575c:	0f 29 84 24 a0 00 00 00 	movaps %xmm0,0xa0(%rsp)
   85764:	4c 8d ac 24 a0 00 00 00 	lea    0xa0(%rsp),%r13
   8576c:	4c 89 ef             	mov    %r13,%rdi
   8576f:	e8 00 00 00 00       	call   85774 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xa4>	85770: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   85774:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
   85779:	48 8d 84 24 b8 00 00 00 	lea    0xb8(%rsp),%rax
   85781:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
   85786:	48 8d 84 24 d0 00 00 00 	lea    0xd0(%rsp),%rax
   8578e:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
   85793:	48 c7 44 24 68 03 00 00 00 	movq   $0x3,0x68(%rsp)
   8579c:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
   857a1:	4c 89 e6             	mov    %r12,%rsi
   857a4:	e8 00 00 00 00       	call   857a9 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xd9>	857a5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   857a9:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 857b0 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xe0>	857ac: R_X86_64_PC32	.bss+0x1f64
   857b0:	84 c0                	test   %al,%al
   857b2:	0f 84 f6 00 00 00    	je     858ae <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1de>
   857b8:	4c 8b 25 00 00 00 00 	mov    0x0(%rip),%r12        # 857bf <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xef>	857bb: R_X86_64_PC32	.bss+0x1f5c
   857bf:	c6 84 24 f0 00 00 00 00 	movb   $0x0,0xf0(%rsp)
   857c7:	c6 84 24 fa 00 00 00 00 	movb   $0x0,0xfa(%rsp)
   857cf:	c6 84 24 fc 00 00 00 00 	movb   $0x0,0xfc(%rsp)
   857d7:	c6 84 24 04 01 00 00 00 	movb   $0x0,0x104(%rsp)
   857df:	c6 84 24 0c 01 00 00 00 	movb   $0x0,0x10c(%rsp)
   857e7:	c6 84 24 14 01 00 00 00 	movb   $0x0,0x114(%rsp)
   857ef:	c6 84 24 1c 01 00 00 00 	movb   $0x0,0x11c(%rsp)
   857f7:	c6 84 24 24 01 00 00 00 	movb   $0x0,0x124(%rsp)
   857ff:	c6 84 24 2c 01 00 00 00 	movb   $0x0,0x12c(%rsp)
   85807:	c6 84 24 38 01 00 00 00 	movb   $0x0,0x138(%rsp)
   8580f:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   85817:	c6 84 24 70 01 00 00 00 	movb   $0x0,0x170(%rsp)
   8581f:	c6 84 24 78 01 00 00 01 	movb   $0x1,0x178(%rsp)
   85827:	48 c7 44 24 40 00 00 00 00 	movq   $0x0,0x40(%rsp)	8582c: R_X86_64_32S	.rodata+0x48488
   85830:	4c 89 7c 24 48       	mov    %r15,0x48(%rsp)
   85835:	0f 57 c0             	xorps  %xmm0,%xmm0
   85838:	0f 29 84 24 90 00 00 00 	movaps %xmm0,0x90(%rsp)
   85840:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
   85845:	e8 00 00 00 00       	call   8584a <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x17a>	85846: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   8584a:	31 c9                	xor    %ecx,%ecx
   8584c:	80 7c 24 28 00       	cmpb   $0x0,0x28(%rsp)
   85851:	0f 94 c1             	sete   %cl
   85854:	0f 28 84 24 90 00 00 00 	movaps 0x90(%rsp),%xmm0
   8585c:	0f 11 04 24          	movups %xmm0,(%rsp)
   85860:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   85864:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   85869:	48 8d 54 24 50       	lea    0x50(%rsp),%rdx
   8586e:	48 8d 4c 24 40       	lea    0x40(%rsp),%rcx
   85873:	4c 8d 8c 24 e8 00 00 00 	lea    0xe8(%rsp),%r9
   8587b:	4c 89 f7             	mov    %r14,%rdi
   8587e:	48 89 de             	mov    %rbx,%rsi
   85881:	4d 89 e0             	mov    %r12,%r8
   85884:	e8 00 00 00 00       	call   85889 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1b9>	85885: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   85889:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
   8588e:	e8 00 00 00 00       	call   85893 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1c3>	8588f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   85893:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   85898:	e8 00 00 00 00       	call   8589d <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1cd>	85899: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   8589d:	48 81 c4 80 01 00 00 	add    $0x180,%rsp
   858a4:	5b                   	pop    %rbx
   858a5:	41 5c                	pop    %r12
   858a7:	41 5d                	pop    %r13
   858a9:	41 5e                	pop    %r14
   858ab:	41 5f                	pop    %r15
   858ad:	c3                   	ret
   858ae:	bf 00 00 00 00       	mov    $0x0,%edi	858af: R_X86_64_32	.bss+0x1f68
   858b3:	e8 00 00 00 00       	call   858b8 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1e8>	858b4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   858b8:	85 c0                	test   %eax,%eax
   858ba:	0f 84 f8 fe ff ff    	je     857b8 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xe8>
   858c0:	bf 00 00 00 00       	mov    $0x0,%edi	858c1: R_X86_64_32	.rodata+0x48420
   858c5:	e8 00 00 00 00       	call   858ca <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x1fa>	858c6: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   858ca:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 858d1 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x201>	858cd: R_X86_64_PC32	.bss+0x1f5c
   858d1:	bf 00 00 00 00       	mov    $0x0,%edi	858d2: R_X86_64_32	.bss+0x1f68
   858d6:	e8 00 00 00 00       	call   858db <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x20b>	858d7: R_X86_64_PLT32	__cxa_guard_release-0x4
   858db:	e9 d8 fe ff ff       	jmp    857b8 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0xe8>
   858e0:	48 89 c3             	mov    %rax,%rbx
   858e3:	bf 00 00 00 00       	mov    $0x0,%edi	858e4: R_X86_64_32	.bss+0x1f68
   858e8:	e8 00 00 00 00       	call   858ed <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x21d>	858e9: R_X86_64_PLT32	__cxa_guard_abort-0x4
   858ed:	eb 03                	jmp    858f2 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x222>
   858ef:	48 89 c3             	mov    %rax,%rbx
   858f2:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
   858f7:	e8 00 00 00 00       	call   858fc <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x22c>	858f8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   858fc:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   85901:	e8 00 00 00 00       	call   85906 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x236>	85902: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   85906:	48 89 df             	mov    %rbx,%rdi
   85909:	e8 00 00 00 00       	call   8590e <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x23e>	8590a: R_X86_64_PLT32	_Unwind_Resume-0x4
   8590e:	48 89 c3             	mov    %rax,%rbx
   85911:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   85916:	e8 00 00 00 00       	call   8591b <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x24b>	85917: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   8591b:	48 89 df             	mov    %rbx,%rdi
   8591e:	e8 00 00 00 00       	call   85923 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_121native_mmvq_q6k_a4_c6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE8_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_+0x253>	8591f: R_X86_64_PLT32	_Unwind_Resume-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text._ZN4dpct10device_ext12create_queueEb:

Disassembly of section .text._ZNK4sycl3_V13vecIfLi2EE7convertINS0_6detail9half_impl4halfELNS_13rounding_modeE1EEENS1_IT_Li2EEEv:

Disassembly of section .text._ZN4sycl3_V15eventD2Ev:

Disassembly of section .text._ZNSt6vectorIfSaIfEEC2EmRKS0_:

Disassembly of section .text._ZN4sycl3_V15queue4waitERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN4dpct12sync_barrierEPN4sycl3_V15eventEPNS1_5queueE:

Disassembly of section .text._ZNSt6vectorIfSaIfEE9push_backEOf:

Disassembly of section .text._ZSt4sortIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEEEvT_S7_:

Disassembly of section .text._ZN4dpct10device_ext13destroy_queueERPN4sycl3_V15queueE:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZN4dpct22has_capability_or_failERKN4sycl3_V16deviceERKSt16initializer_listINS1_6aspectEE:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

Disassembly of section .text._ZN4dpct7dev_mgr14current_deviceEv:

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

Disassembly of section .text._ZN4dpct10device_ext17create_queue_implIJN4sycl3_V18property5queue8in_orderEEEEPNS3_5queueEbDpT_:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EE9push_backEOS4_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZN4sycl3_V114exception_listD2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZN4sycl3_V15queueD2Ev:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEElNS0_5__ops15_Iter_less_iterEEvT_S9_T0_T1_:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V15queueELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_:

Disassembly of section .text._ZNK4sycl3_V16device8get_infoINS0_4info6device4nameEEENS0_6detail19is_device_info_descIT_E11return_typeEv:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_S9_:

Disassembly of section .text._ZZN4dpct22has_capability_or_failERKN4sycl3_V16deviceERKSt16initializer_listINS1_6aspectEEENKUlS6_E_clB5cxx11ES6_:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7replaceEmmPKcm:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_:

Disassembly of section .text._ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail19defaultAsyncHandlerENS0_14exception_listE:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEPS3_E9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEPS3_E10_M_managerERSt9_Any_dataRKS6_St18_Manager_operation:

Disassembly of section .text._ZN4dpct6detail7mem_mgrC2Ev:

Disassembly of section .text._ZN4dpct6detail7mem_mgrD2Ev:

Disassembly of section .text._ZNSt8_Rb_treeIPhSt4pairIKS0_N4dpct6detail7mem_mgr10allocationEESt10_Select1stIS7_ESt4lessIS0_ESaIS7_EE8_M_eraseEPSt13_Rb_tree_nodeIS7_E:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels34native_quantize_q8_1_kernel_b480d2EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels41native_swiglu_quantize_q8_1_kernel_85e13eEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q5_k_mmvq_kernel_b1d5a2E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q5_k_mmvq_kernel_938b09E18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q2_0_mmvq_kernel_dec305E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q2_0_mmvq_kernel_ef5f5cE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q3_k_mmvq_kernel_208be9E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q3_k_mmvq_kernel_926817E18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4dpct6detail13device_memoryIaLNS_13memory_regionE0ELm1EE12get_ptr_implERN4sycl3_V15queueE:

Disassembly of section .text._ZNSt6vectorIN4sycl3_V15eventESaIS2_EED2Ev:

Disassembly of section .text._ZN4dpct6detail13device_memoryIaLNS_13memory_regionE0ELm1EE7get_ptrEv:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32native_iq4_xs_mmvq_kernel_22a688E18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels32native_iq4_xs_mmvq_kernel_3ba00dE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q4_k_mmvq_kernel_da128dE18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q4_k_mmvq_kernel_b29f4fE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q6_k_mmvq_kernel_4d887fE18dpct_kernel_scalarILi1EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN6strata7kernels30native_q6_k_mmvq_kernel_19484eE18dpct_kernel_scalarILi0EEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:
