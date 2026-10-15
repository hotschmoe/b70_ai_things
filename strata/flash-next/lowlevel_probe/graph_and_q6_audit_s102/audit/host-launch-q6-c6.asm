
/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

000000000004c670 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE>:
   4c670:	55                   	push   %rbp
   4c671:	41 57                	push   %r15
   4c673:	41 56                	push   %r14
   4c675:	41 55                	push   %r13
   4c677:	41 54                	push   %r12
   4c679:	53                   	push   %rbx
   4c67a:	48 81 ec c8 04 00 00 	sub    $0x4c8,%rsp
   4c681:	4c 89 cb             	mov    %r9,%rbx
   4c684:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c68b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b>
			4c687: R_X86_64_PC32	.bss+0x1944
   4c68b:	84 c0                	test   %al,%al
   4c68d:	0f 84 05 28 00 00    	je     4ee98 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2828>
   4c693:	8b 2d 00 00 00 00    	mov    0x0(%rip),%ebp        # 4c699 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29>
			4c695: R_X86_64_PC32	.bss+0x193c
   4c699:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c6a0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30>
			4c69c: R_X86_64_PC32	.bss+0x1954
   4c6a0:	84 c0                	test   %al,%al
   4c6a2:	0f 84 40 28 00 00    	je     4eee8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2878>
   4c6a8:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 4c6ae <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3e>
			4c6aa: R_X86_64_PC32	.bss+0x194c
   4c6ae:	85 c0                	test   %eax,%eax
   4c6b0:	0f 8e 95 02 00 00    	jle    4c94b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2db>
   4c6b6:	83 c0 fe             	add    $0xfffffffe,%eax
   4c6b9:	83 f8 04             	cmp    $0x4,%eax
   4c6bc:	0f 87 89 02 00 00    	ja     4c94b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2db>
   4c6c2:	4d 63 f8             	movslq %r8d,%r15
   4c6c5:	49 83 c7 03          	add    $0x3,%r15
   4c6c9:	ff 24 c5 00 00 00 00 	jmp    *0x0(,%rax,8)
			4c6cc: R_X86_64_32S	.rodata+0x320
   4c6d0:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c6d7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x67>
			4c6d3: R_X86_64_PC32	.bss+0x204
   4c6d7:	84 c0                	test   %al,%al
   4c6d9:	0f 84 0e 29 00 00    	je     4efed <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x297d>
   4c6df:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4c6e6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x76>
			4c6e1: R_X86_64_PC32	.bss+0x1fb
   4c6e6:	0f 84 d4 15 00 00    	je     4dcc0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1650>
   4c6ec:	0f 57 c0             	xorps  %xmm0,%xmm0
   4c6ef:	0f 29 84 24 90 03 00 	movaps %xmm0,0x390(%rsp)
   4c6f6:	00 
   4c6f7:	48 c7 84 24 a0 03 00 	movq   $0x0,0x3a0(%rsp)
   4c6fe:	00 00 00 00 00 
   4c703:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4c708:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4c70d:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4c712:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4c719:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4c720:	00 
   4c721:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4c728 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xb8>
			4c724: R_X86_64_PC32	.rodata+0x78464
   4c728:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4c72f:	00 
   4c730:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4c737 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xc7>
			4c733: R_X86_64_PC32	.rodata+0x78454
   4c737:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4c73e:	00 
   4c73f:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4c746:	00 
   4c747:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4c74e:	00 
   4c74f:	4c 89 f7             	mov    %r14,%rdi
   4c752:	e8 00 00 00 00       	call   4c757 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe7>
			4c753: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c757:	4c 89 f7             	mov    %r14,%rdi
   4c75a:	e8 00 00 00 00       	call   4c75f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xef>
			4c75b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4c75f:	49 89 c6             	mov    %rax,%r14
   4c762:	49 c1 e7 03          	shl    $0x3,%r15
   4c766:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4c76d:	00 00 00 
   4c770:	4c 21 f8             	and    %r15,%rax
   4c773:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4c77a:	00 01 00 00 00 
   4c77f:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4c786:	00 04 00 00 00 
   4c78b:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4c792:	00 
   4c793:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4c79a:	00 01 00 00 00 
   4c79f:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4c7a6:	00 04 00 00 00 
   4c7ab:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4c7b2:	00 20 00 00 00 
   4c7b7:	0f 28 84 24 90 03 00 	movaps 0x390(%rsp),%xmm0
   4c7be:	00 
   4c7bf:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4c7c6:	00 
   4c7c7:	48 8b 84 24 a0 03 00 	mov    0x3a0(%rsp),%rax
   4c7ce:	00 
   4c7cf:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4c7d6:	00 
   4c7d7:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4c7de:	00 
   4c7df:	4c 89 ff             	mov    %r15,%rdi
   4c7e2:	e8 00 00 00 00       	call   4c7e7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x177>
			4c7e3: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4c7e7:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4c7ee:	00 
   4c7ef:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4c7f6:	00 
   4c7f7:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4c7fe:	00 
   4c7ff:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4c806:	00 
   4c807:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4c80e:	00 
   4c80f:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4c816:	00 03 00 00 00 
   4c81b:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c820:	4c 89 f6             	mov    %r14,%rsi
   4c823:	e8 00 00 00 00       	call   4c828 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b8>
			4c824: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c828:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c82f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1bf>
			4c82b: R_X86_64_PC32	.bss+0x1eb4
   4c82f:	84 c0                	test   %al,%al
   4c831:	0f 84 b1 2a 00 00    	je     4f2e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c78>
   4c837:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4c83e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1ce>
			4c83a: R_X86_64_PC32	.bss+0x1eac
   4c83e:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4c845:	00 
   4c846:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4c84d:	00 
   4c84e:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4c855:	00 
   4c856:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4c85d:	00 
   4c85e:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4c865:	00 
   4c866:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4c86d:	00 
   4c86e:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4c875:	00 
   4c876:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4c87d:	00 
   4c87e:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4c885:	00 
   4c886:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4c88d:	00 
   4c88e:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4c895:	00 
   4c896:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4c89d:	00 
   4c89e:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4c8a5:	01 
   4c8a6:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4c8ad:	00 00 00 00 00 
			4c8ae: R_X86_64_32S	.rodata+0x47200
   4c8b2:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4c8b7:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4c8be:	00 
   4c8bf:	0f 57 c0             	xorps  %xmm0,%xmm0
   4c8c2:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4c8c9:	00 
   4c8ca:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c8cf:	e8 00 00 00 00       	call   4c8d4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x264>
			4c8d0: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4c8d4:	31 c9                	xor    %ecx,%ecx
   4c8d6:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4c8db:	0f 94 c1             	sete   %cl
   4c8de:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4c8e5:	00 
   4c8e6:	0f 11 04 24          	movups %xmm0,(%rsp)
   4c8ea:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4c8ee:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4c8f3:	48 8d bc 24 88 04 00 	lea    0x488(%rsp),%rdi
   4c8fa:	00 
   4c8fb:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4c902:	00 
   4c903:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4c90a:	00 
   4c90b:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4c912:	00 
   4c913:	48 89 de             	mov    %rbx,%rsi
   4c916:	4d 89 f0             	mov    %r14,%r8
   4c919:	e8 00 00 00 00       	call   4c91e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ae>
			4c91a: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4c91e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c923:	e8 00 00 00 00       	call   4c928 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b8>
			4c924: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4c928:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4c92f:	00 
   4c930:	e8 00 00 00 00       	call   4c935 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c5>
			4c931: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4c935:	48 8b 9c 24 90 04 00 	mov    0x490(%rsp),%rbx
   4c93c:	00 
   4c93d:	48 85 db             	test   %rbx,%rbx
   4c940:	0f 85 89 21 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4c946:	e9 df 21 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4c94b:	83 fd 04             	cmp    $0x4,%ebp
   4c94e:	0f 84 63 0c 00 00    	je     4d5b7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf47>
   4c954:	83 fd 02             	cmp    $0x2,%ebp
   4c957:	0f 84 f8 09 00 00    	je     4d355 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xce5>
   4c95d:	83 fd 01             	cmp    $0x1,%ebp
   4c960:	0f 85 b2 0e 00 00    	jne    4d818 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11a8>
   4c966:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c96d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fd>
			4c969: R_X86_64_PC32	.bss+0x204
   4c96d:	84 c0                	test   %al,%al
   4c96f:	0f 84 58 28 00 00    	je     4f1cd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b5d>
   4c975:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4c97c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30c>
			4c977: R_X86_64_PC32	.bss+0x1fb
   4c97c:	0f 84 d8 1e 00 00    	je     4e85a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21ea>
   4c982:	4d 63 f8             	movslq %r8d,%r15
   4c985:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c98c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31c>
			4c988: R_X86_64_PC32	.bss+0x1954
   4c98c:	84 c0                	test   %al,%al
   4c98e:	0f 84 c7 28 00 00    	je     4f25b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2beb>
   4c994:	49 83 c7 03          	add    $0x3,%r15
   4c998:	83 3d 00 00 00 00 00 	cmpl   $0x0,0x0(%rip)        # 4c99f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x32f>
			4c99a: R_X86_64_PC32	.bss+0x194b
   4c99f:	0f 84 97 21 00 00    	je     4eb3c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24cc>
   4c9a5:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4c9aa:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4c9af:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4c9b4:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4c9bb:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4c9c2:	00 
   4c9c3:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4c9ca <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x35a>
			4c9c6: R_X86_64_PC32	.rodata+0x78464
   4c9ca:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4c9d1:	00 
   4c9d2:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4c9d9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x369>
			4c9d5: R_X86_64_PC32	.rodata+0x78454
   4c9d9:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4c9e0:	00 
   4c9e1:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4c9e8:	00 
   4c9e9:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4c9f0:	00 
   4c9f1:	4c 89 f7             	mov    %r14,%rdi
   4c9f4:	e8 00 00 00 00       	call   4c9f9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x389>
			4c9f5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c9f9:	4c 89 f7             	mov    %r14,%rdi
   4c9fc:	e8 00 00 00 00       	call   4ca01 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x391>
			4c9fd: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4ca01:	49 89 c6             	mov    %rax,%r14
   4ca04:	49 c1 e7 03          	shl    $0x3,%r15
   4ca08:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4ca0f:	00 00 00 
   4ca12:	4c 21 f8             	and    %r15,%rax
   4ca15:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4ca1c:	00 01 00 00 00 
   4ca21:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4ca28:	00 04 00 00 00 
   4ca2d:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4ca34:	00 
   4ca35:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4ca3c:	00 01 00 00 00 
   4ca41:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4ca48:	00 04 00 00 00 
   4ca4d:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4ca54:	00 20 00 00 00 
   4ca59:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ca5c:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4ca63:	00 
   4ca64:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4ca6b:	00 00 00 00 00 
   4ca70:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4ca77:	00 
   4ca78:	4c 89 ff             	mov    %r15,%rdi
   4ca7b:	e8 00 00 00 00       	call   4ca80 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x410>
			4ca7c: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4ca80:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4ca87:	00 
   4ca88:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4ca8f:	00 
   4ca90:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4ca97:	00 
   4ca98:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4ca9f:	00 
   4caa0:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4caa7:	00 
   4caa8:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4caaf:	00 03 00 00 00 
   4cab4:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cab9:	4c 89 f6             	mov    %r14,%rsi
   4cabc:	e8 00 00 00 00       	call   4cac1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x451>
			4cabd: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cac1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cac8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x458>
			4cac4: R_X86_64_PC32	.bss+0x1f54
   4cac8:	84 c0                	test   %al,%al
   4caca:	0f 84 12 29 00 00    	je     4f3e2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d72>
   4cad0:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cad7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x467>
			4cad3: R_X86_64_PC32	.bss+0x1f4c
   4cad7:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4cade:	00 
   4cadf:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4cae6:	00 
   4cae7:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4caee:	00 
   4caef:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4caf6:	00 
   4caf7:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4cafe:	00 
   4caff:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4cb06:	00 
   4cb07:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4cb0e:	00 
   4cb0f:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4cb16:	00 
   4cb17:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4cb1e:	00 
   4cb1f:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4cb26:	00 
   4cb27:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4cb2e:	00 
   4cb2f:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4cb36:	00 
   4cb37:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4cb3e:	01 
   4cb3f:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4cb46:	00 00 00 00 00 
			4cb47: R_X86_64_32S	.rodata+0x48448
   4cb4b:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4cb50:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4cb57:	00 
   4cb58:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cb5b:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4cb62:	00 
   4cb63:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cb68:	e8 00 00 00 00       	call   4cb6d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x4fd>
			4cb69: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cb6d:	31 c9                	xor    %ecx,%ecx
   4cb6f:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4cb74:	0f 94 c1             	sete   %cl
   4cb77:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4cb7e:	00 
   4cb7f:	0f 11 04 24          	movups %xmm0,(%rsp)
   4cb83:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4cb87:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4cb8c:	48 8d bc 24 08 04 00 	lea    0x408(%rsp),%rdi
   4cb93:	00 
   4cb94:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4cb9b:	00 
   4cb9c:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4cba3:	00 
   4cba4:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4cbab:	00 
   4cbac:	48 89 de             	mov    %rbx,%rsi
   4cbaf:	4d 89 f0             	mov    %r14,%r8
   4cbb2:	e8 00 00 00 00       	call   4cbb7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x547>
			4cbb3: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4cbb7:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cbbc:	e8 00 00 00 00       	call   4cbc1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x551>
			4cbbd: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4cbc1:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4cbc8:	00 
   4cbc9:	e8 00 00 00 00       	call   4cbce <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x55e>
			4cbca: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4cbce:	48 8b 9c 24 10 04 00 	mov    0x410(%rsp),%rbx
   4cbd5:	00 
   4cbd6:	48 85 db             	test   %rbx,%rbx
   4cbd9:	0f 85 f0 1e 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4cbdf:	e9 46 1f 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4cbe4:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cbeb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x57b>
			4cbe7: R_X86_64_PC32	.bss+0x204
   4cbeb:	84 c0                	test   %al,%al
   4cbed:	0f 84 9a 23 00 00    	je     4ef8d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x291d>
   4cbf3:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4cbfa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x58a>
			4cbf5: R_X86_64_PC32	.bss+0x1fb
   4cbfa:	0f 84 61 0e 00 00    	je     4da61 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13f1>
   4cc00:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cc03:	0f 29 84 24 10 03 00 	movaps %xmm0,0x310(%rsp)
   4cc0a:	00 
   4cc0b:	48 c7 84 24 20 03 00 	movq   $0x0,0x320(%rsp)
   4cc12:	00 00 00 00 00 
   4cc17:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4cc1c:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4cc21:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4cc26:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4cc2d:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4cc34:	00 
   4cc35:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4cc3c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5cc>
			4cc38: R_X86_64_PC32	.rodata+0x78464
   4cc3c:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4cc43:	00 
   4cc44:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4cc4b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5db>
			4cc47: R_X86_64_PC32	.rodata+0x78454
   4cc4b:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4cc52:	00 
   4cc53:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4cc5a:	00 
   4cc5b:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4cc62:	00 
   4cc63:	4c 89 f7             	mov    %r14,%rdi
   4cc66:	e8 00 00 00 00       	call   4cc6b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5fb>
			4cc67: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cc6b:	4c 89 f7             	mov    %r14,%rdi
   4cc6e:	e8 00 00 00 00       	call   4cc73 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x603>
			4cc6f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cc73:	49 89 c6             	mov    %rax,%r14
   4cc76:	49 c1 e7 03          	shl    $0x3,%r15
   4cc7a:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4cc81:	00 00 00 
   4cc84:	4c 21 f8             	and    %r15,%rax
   4cc87:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4cc8e:	00 01 00 00 00 
   4cc93:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4cc9a:	00 04 00 00 00 
   4cc9f:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4cca6:	00 
   4cca7:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4ccae:	00 01 00 00 00 
   4ccb3:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4ccba:	00 04 00 00 00 
   4ccbf:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4ccc6:	00 20 00 00 00 
   4cccb:	0f 28 84 24 10 03 00 	movaps 0x310(%rsp),%xmm0
   4ccd2:	00 
   4ccd3:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4ccda:	00 
   4ccdb:	48 8b 84 24 20 03 00 	mov    0x320(%rsp),%rax
   4cce2:	00 
   4cce3:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4ccea:	00 
   4cceb:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4ccf2:	00 
   4ccf3:	4c 89 ff             	mov    %r15,%rdi
   4ccf6:	e8 00 00 00 00       	call   4ccfb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x68b>
			4ccf7: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4ccfb:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4cd02:	00 
   4cd03:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4cd0a:	00 
   4cd0b:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4cd12:	00 
   4cd13:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4cd1a:	00 
   4cd1b:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4cd22:	00 
   4cd23:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4cd2a:	00 03 00 00 00 
   4cd2f:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cd34:	4c 89 f6             	mov    %r14,%rsi
   4cd37:	e8 00 00 00 00       	call   4cd3c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6cc>
			4cd38: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cd3c:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cd43 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6d3>
			4cd3f: R_X86_64_PC32	.bss+0x1ef4
   4cd43:	84 c0                	test   %al,%al
   4cd45:	0f 84 6b 25 00 00    	je     4f2b6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c46>
   4cd4b:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cd52 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6e2>
			4cd4e: R_X86_64_PC32	.bss+0x1eec
   4cd52:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4cd59:	00 
   4cd5a:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4cd61:	00 
   4cd62:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4cd69:	00 
   4cd6a:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4cd71:	00 
   4cd72:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4cd79:	00 
   4cd7a:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4cd81:	00 
   4cd82:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4cd89:	00 
   4cd8a:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4cd91:	00 
   4cd92:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4cd99:	00 
   4cd9a:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4cda1:	00 
   4cda2:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4cda9:	00 
   4cdaa:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4cdb1:	00 
   4cdb2:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4cdb9:	01 
   4cdba:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4cdc1:	00 00 00 00 00 
			4cdc2: R_X86_64_32S	.rodata+0x47a20
   4cdc6:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4cdcb:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4cdd2:	00 
   4cdd3:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cdd6:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4cddd:	00 
   4cdde:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cde3:	e8 00 00 00 00       	call   4cde8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x778>
			4cde4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cde8:	31 c9                	xor    %ecx,%ecx
   4cdea:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4cdef:	0f 94 c1             	sete   %cl
   4cdf2:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4cdf9:	00 
   4cdfa:	0f 11 04 24          	movups %xmm0,(%rsp)
   4cdfe:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4ce02:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4ce07:	48 8d bc 24 48 04 00 	lea    0x448(%rsp),%rdi
   4ce0e:	00 
   4ce0f:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4ce16:	00 
   4ce17:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4ce1e:	00 
   4ce1f:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4ce26:	00 
   4ce27:	48 89 de             	mov    %rbx,%rsi
   4ce2a:	4d 89 f0             	mov    %r14,%r8
   4ce2d:	e8 00 00 00 00       	call   4ce32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7c2>
			4ce2e: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4ce32:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ce37:	e8 00 00 00 00       	call   4ce3c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7cc>
			4ce38: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4ce3c:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4ce43:	00 
   4ce44:	e8 00 00 00 00       	call   4ce49 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7d9>
			4ce45: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4ce49:	48 8b 9c 24 50 04 00 	mov    0x450(%rsp),%rbx
   4ce50:	00 
   4ce51:	48 85 db             	test   %rbx,%rbx
   4ce54:	0f 85 75 1c 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4ce5a:	e9 cb 1c 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4ce5f:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4ce66 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7f6>
			4ce62: R_X86_64_PC32	.bss+0x204
   4ce66:	84 c0                	test   %al,%al
   4ce68:	0f 84 df 21 00 00    	je     4f04d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29dd>
   4ce6e:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4ce75 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x805>
			4ce70: R_X86_64_PC32	.bss+0x1fb
   4ce75:	0f 84 a4 10 00 00    	je     4df1f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x18af>
   4ce7b:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ce7e:	0f 29 84 24 50 03 00 	movaps %xmm0,0x350(%rsp)
   4ce85:	00 
   4ce86:	48 c7 84 24 60 03 00 	movq   $0x0,0x360(%rsp)
   4ce8d:	00 00 00 00 00 
   4ce92:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4ce97:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4ce9c:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4cea1:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4cea8:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4ceaf:	00 
   4ceb0:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4ceb7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x847>
			4ceb3: R_X86_64_PC32	.rodata+0x78464
   4ceb7:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4cebe:	00 
   4cebf:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4cec6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x856>
			4cec2: R_X86_64_PC32	.rodata+0x78454
   4cec6:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4cecd:	00 
   4cece:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4ced5:	00 
   4ced6:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4cedd:	00 
   4cede:	4c 89 f7             	mov    %r14,%rdi
   4cee1:	e8 00 00 00 00       	call   4cee6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x876>
			4cee2: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cee6:	4c 89 f7             	mov    %r14,%rdi
   4cee9:	e8 00 00 00 00       	call   4ceee <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x87e>
			4ceea: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4ceee:	49 89 c6             	mov    %rax,%r14
   4cef1:	49 c1 e7 03          	shl    $0x3,%r15
   4cef5:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4cefc:	00 00 00 
   4ceff:	4c 21 f8             	and    %r15,%rax
   4cf02:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4cf09:	00 01 00 00 00 
   4cf0e:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4cf15:	00 04 00 00 00 
   4cf1a:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4cf21:	00 
   4cf22:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4cf29:	00 01 00 00 00 
   4cf2e:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4cf35:	00 04 00 00 00 
   4cf3a:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4cf41:	00 20 00 00 00 
   4cf46:	0f 28 84 24 50 03 00 	movaps 0x350(%rsp),%xmm0
   4cf4d:	00 
   4cf4e:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4cf55:	00 
   4cf56:	48 8b 84 24 60 03 00 	mov    0x360(%rsp),%rax
   4cf5d:	00 
   4cf5e:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4cf65:	00 
   4cf66:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4cf6d:	00 
   4cf6e:	4c 89 ff             	mov    %r15,%rdi
   4cf71:	e8 00 00 00 00       	call   4cf76 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x906>
			4cf72: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4cf76:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4cf7d:	00 
   4cf7e:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4cf85:	00 
   4cf86:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4cf8d:	00 
   4cf8e:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4cf95:	00 
   4cf96:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4cf9d:	00 
   4cf9e:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4cfa5:	00 03 00 00 00 
   4cfaa:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cfaf:	4c 89 f6             	mov    %r14,%rsi
   4cfb2:	e8 00 00 00 00       	call   4cfb7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x947>
			4cfb3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cfb7:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cfbe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x94e>
			4cfba: R_X86_64_PC32	.bss+0x1ed4
   4cfbe:	84 c0                	test   %al,%al
   4cfc0:	0f 84 54 23 00 00    	je     4f31a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2caa>
   4cfc6:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cfcd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x95d>
			4cfc9: R_X86_64_PC32	.bss+0x1ecc
   4cfcd:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4cfd4:	00 
   4cfd5:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4cfdc:	00 
   4cfdd:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4cfe4:	00 
   4cfe5:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4cfec:	00 
   4cfed:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4cff4:	00 
   4cff5:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4cffc:	00 
   4cffd:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4d004:	00 
   4d005:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4d00c:	00 
   4d00d:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4d014:	00 
   4d015:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4d01c:	00 
   4d01d:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4d024:	00 
   4d025:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4d02c:	00 
   4d02d:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4d034:	01 
   4d035:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4d03c:	00 00 00 00 00 
			4d03d: R_X86_64_32S	.rodata+0x47610
   4d041:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d046:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4d04d:	00 
   4d04e:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d051:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4d058:	00 
   4d059:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d05e:	e8 00 00 00 00       	call   4d063 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x9f3>
			4d05f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d063:	31 c9                	xor    %ecx,%ecx
   4d065:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d06a:	0f 94 c1             	sete   %cl
   4d06d:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4d074:	00 
   4d075:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d079:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d07d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d082:	48 8d bc 24 68 04 00 	lea    0x468(%rsp),%rdi
   4d089:	00 
   4d08a:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4d091:	00 
   4d092:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4d099:	00 
   4d09a:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4d0a1:	00 
   4d0a2:	48 89 de             	mov    %rbx,%rsi
   4d0a5:	4d 89 f0             	mov    %r14,%r8
   4d0a8:	e8 00 00 00 00       	call   4d0ad <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa3d>
			4d0a9: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d0ad:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d0b2:	e8 00 00 00 00       	call   4d0b7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa47>
			4d0b3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d0b7:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4d0be:	00 
   4d0bf:	e8 00 00 00 00       	call   4d0c4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa54>
			4d0c0: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d0c4:	48 8b 9c 24 70 04 00 	mov    0x470(%rsp),%rbx
   4d0cb:	00 
   4d0cc:	48 85 db             	test   %rbx,%rbx
   4d0cf:	0f 85 fa 19 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4d0d5:	e9 50 1a 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4d0da:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d0e1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa71>
			4d0dd: R_X86_64_PC32	.bss+0x204
   4d0e1:	84 c0                	test   %al,%al
   4d0e3:	0f 84 c4 1f 00 00    	je     4f0ad <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a3d>
   4d0e9:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d0f0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa80>
			4d0eb: R_X86_64_PC32	.bss+0x1fb
   4d0f0:	0f 84 88 10 00 00    	je     4e17e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b0e>
   4d0f6:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d0f9:	0f 29 84 24 d0 02 00 	movaps %xmm0,0x2d0(%rsp)
   4d100:	00 
   4d101:	48 c7 84 24 e0 02 00 	movq   $0x0,0x2e0(%rsp)
   4d108:	00 00 00 00 00 
   4d10d:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d112:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d117:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d11c:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d123:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4d12a:	00 
   4d12b:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d132 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xac2>
			4d12e: R_X86_64_PC32	.rodata+0x78464
   4d132:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4d139:	00 
   4d13a:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d141 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xad1>
			4d13d: R_X86_64_PC32	.rodata+0x78454
   4d141:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4d148:	00 
   4d149:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4d150:	00 
   4d151:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4d158:	00 
   4d159:	4c 89 f7             	mov    %r14,%rdi
   4d15c:	e8 00 00 00 00       	call   4d161 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xaf1>
			4d15d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d161:	4c 89 f7             	mov    %r14,%rdi
   4d164:	e8 00 00 00 00       	call   4d169 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xaf9>
			4d165: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d169:	49 89 c6             	mov    %rax,%r14
   4d16c:	49 c1 e7 03          	shl    $0x3,%r15
   4d170:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4d177:	00 00 00 
   4d17a:	4c 21 f8             	and    %r15,%rax
   4d17d:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4d184:	00 01 00 00 00 
   4d189:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4d190:	00 04 00 00 00 
   4d195:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4d19c:	00 
   4d19d:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4d1a4:	00 01 00 00 00 
   4d1a9:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4d1b0:	00 04 00 00 00 
   4d1b5:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4d1bc:	00 20 00 00 00 
   4d1c1:	0f 28 84 24 d0 02 00 	movaps 0x2d0(%rsp),%xmm0
   4d1c8:	00 
   4d1c9:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4d1d0:	00 
   4d1d1:	48 8b 84 24 e0 02 00 	mov    0x2e0(%rsp),%rax
   4d1d8:	00 
   4d1d9:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4d1e0:	00 
   4d1e1:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4d1e8:	00 
   4d1e9:	4c 89 ff             	mov    %r15,%rdi
   4d1ec:	e8 00 00 00 00       	call   4d1f1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xb81>
			4d1ed: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d1f1:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4d1f8:	00 
   4d1f9:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4d200:	00 
   4d201:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4d208:	00 
   4d209:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4d210:	00 
   4d211:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4d218:	00 
   4d219:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4d220:	00 03 00 00 00 
   4d225:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d22a:	4c 89 f6             	mov    %r14,%rsi
   4d22d:	e8 00 00 00 00       	call   4d232 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbc2>
			4d22e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d232:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d239 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbc9>
			4d235: R_X86_64_PC32	.bss+0x1f14
   4d239:	84 c0                	test   %al,%al
   4d23b:	0f 84 0b 21 00 00    	je     4f34c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cdc>
   4d241:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d248 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbd8>
			4d244: R_X86_64_PC32	.bss+0x1f0c
   4d248:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4d24f:	00 
   4d250:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4d257:	00 
   4d258:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4d25f:	00 
   4d260:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4d267:	00 
   4d268:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4d26f:	00 
   4d270:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4d277:	00 
   4d278:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4d27f:	00 
   4d280:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4d287:	00 
   4d288:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4d28f:	00 
   4d290:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4d297:	00 
   4d298:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4d29f:	00 
   4d2a0:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4d2a7:	00 
   4d2a8:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4d2af:	01 
   4d2b0:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4d2b7:	00 00 00 00 00 
			4d2b8: R_X86_64_32S	.rodata+0x47e30
   4d2bc:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d2c1:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4d2c8:	00 
   4d2c9:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d2cc:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4d2d3:	00 
   4d2d4:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d2d9:	e8 00 00 00 00       	call   4d2de <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xc6e>
			4d2da: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d2de:	31 c9                	xor    %ecx,%ecx
   4d2e0:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d2e5:	0f 94 c1             	sete   %cl
   4d2e8:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4d2ef:	00 
   4d2f0:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d2f4:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d2f8:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d2fd:	48 8d bc 24 28 04 00 	lea    0x428(%rsp),%rdi
   4d304:	00 
   4d305:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4d30c:	00 
   4d30d:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4d314:	00 
   4d315:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4d31c:	00 
   4d31d:	48 89 de             	mov    %rbx,%rsi
   4d320:	4d 89 f0             	mov    %r14,%r8
   4d323:	e8 00 00 00 00       	call   4d328 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcb8>
			4d324: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d328:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d32d:	e8 00 00 00 00       	call   4d332 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcc2>
			4d32e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d332:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4d339:	00 
   4d33a:	e8 00 00 00 00       	call   4d33f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xccf>
			4d33b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d33f:	48 8b 9c 24 30 04 00 	mov    0x430(%rsp),%rbx
   4d346:	00 
   4d347:	48 85 db             	test   %rbx,%rbx
   4d34a:	0f 85 7f 17 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4d350:	e9 d5 17 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4d355:	4d 63 f8             	movslq %r8d,%r15
   4d358:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d35f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcef>
			4d35b: R_X86_64_PC32	.bss+0x204
   4d35f:	84 c0                	test   %al,%al
   4d361:	0f 84 a6 1d 00 00    	je     4f10d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a9d>
   4d367:	49 83 c7 07          	add    $0x7,%r15
   4d36b:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d372 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd02>
			4d36d: R_X86_64_PC32	.bss+0x1fb
   4d372:	0f 84 65 10 00 00    	je     4e3dd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d6d>
   4d378:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d37d:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d382:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d387:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d38e:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4d395:	00 
   4d396:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d39d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd2d>
			4d399: R_X86_64_PC32	.rodata+0x78464
   4d39d:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4d3a4:	00 
   4d3a5:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d3ac <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd3c>
			4d3a8: R_X86_64_PC32	.rodata+0x78454
   4d3ac:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4d3b3:	00 
   4d3b4:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4d3bb:	00 
   4d3bc:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4d3c3:	00 
   4d3c4:	4c 89 f7             	mov    %r14,%rdi
   4d3c7:	e8 00 00 00 00       	call   4d3cc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd5c>
			4d3c8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d3cc:	4c 89 f7             	mov    %r14,%rdi
   4d3cf:	e8 00 00 00 00       	call   4d3d4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd64>
			4d3d0: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d3d4:	49 89 c6             	mov    %rax,%r14
   4d3d7:	49 c1 e7 02          	shl    $0x2,%r15
   4d3db:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4d3e2:	00 00 00 
   4d3e5:	4c 21 f8             	and    %r15,%rax
   4d3e8:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4d3ef:	00 01 00 00 00 
   4d3f4:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4d3fb:	00 04 00 00 00 
   4d400:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4d407:	00 
   4d408:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4d40f:	00 01 00 00 00 
   4d414:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4d41b:	00 04 00 00 00 
   4d420:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4d427:	00 20 00 00 00 
   4d42c:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d42f:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4d436:	00 
   4d437:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4d43e:	00 00 00 00 00 
   4d443:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4d44a:	00 
   4d44b:	4c 89 ff             	mov    %r15,%rdi
   4d44e:	e8 00 00 00 00       	call   4d453 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xde3>
			4d44f: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d453:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4d45a:	00 
   4d45b:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4d462:	00 
   4d463:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4d46a:	00 
   4d46b:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4d472:	00 
   4d473:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4d47a:	00 
   4d47b:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4d482:	00 03 00 00 00 
   4d487:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d48c:	4c 89 f6             	mov    %r14,%rsi
   4d48f:	e8 00 00 00 00       	call   4d494 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe24>
			4d490: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d494:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d49b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe2b>
			4d497: R_X86_64_PC32	.bss+0x1f94
   4d49b:	84 c0                	test   %al,%al
   4d49d:	0f 84 db 1e 00 00    	je     4f37e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d0e>
   4d4a3:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d4aa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe3a>
			4d4a6: R_X86_64_PC32	.bss+0x1f8c
   4d4aa:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4d4b1:	00 
   4d4b2:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4d4b9:	00 
   4d4ba:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4d4c1:	00 
   4d4c2:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4d4c9:	00 
   4d4ca:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4d4d1:	00 
   4d4d2:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4d4d9:	00 
   4d4da:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4d4e1:	00 
   4d4e2:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4d4e9:	00 
   4d4ea:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4d4f1:	00 
   4d4f2:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4d4f9:	00 
   4d4fa:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4d501:	00 
   4d502:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4d509:	00 
   4d50a:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4d511:	01 
   4d512:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4d519:	00 00 00 00 00 
			4d51a: R_X86_64_32S	.rodata+0x48c68
   4d51e:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d523:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4d52a:	00 
   4d52b:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d52e:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4d535:	00 
   4d536:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d53b:	e8 00 00 00 00       	call   4d540 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xed0>
			4d53c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d540:	31 c9                	xor    %ecx,%ecx
   4d542:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d547:	0f 94 c1             	sete   %cl
   4d54a:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4d551:	00 
   4d552:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d556:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d55a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d55f:	48 8d bc 24 e8 03 00 	lea    0x3e8(%rsp),%rdi
   4d566:	00 
   4d567:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4d56e:	00 
   4d56f:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4d576:	00 
   4d577:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4d57e:	00 
   4d57f:	48 89 de             	mov    %rbx,%rsi
   4d582:	4d 89 f0             	mov    %r14,%r8
   4d585:	e8 00 00 00 00       	call   4d58a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf1a>
			4d586: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d58a:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d58f:	e8 00 00 00 00       	call   4d594 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf24>
			4d590: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d594:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4d59b:	00 
   4d59c:	e8 00 00 00 00       	call   4d5a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf31>
			4d59d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d5a1:	48 8b 9c 24 f0 03 00 	mov    0x3f0(%rsp),%rbx
   4d5a8:	00 
   4d5a9:	48 85 db             	test   %rbx,%rbx
   4d5ac:	0f 85 1d 15 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4d5b2:	e9 73 15 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4d5b7:	4d 63 f8             	movslq %r8d,%r15
   4d5ba:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d5c1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf51>
			4d5bd: R_X86_64_PC32	.bss+0x204
   4d5c1:	84 c0                	test   %al,%al
   4d5c3:	0f 84 a4 1b 00 00    	je     4f16d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2afd>
   4d5c9:	49 83 c7 0f          	add    $0xf,%r15
   4d5cd:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d5d4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf64>
			4d5cf: R_X86_64_PC32	.bss+0x1fb
   4d5d4:	0f 84 42 10 00 00    	je     4e61c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fac>
   4d5da:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d5df:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d5e4:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d5e9:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d5f0:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4d5f7:	00 
   4d5f8:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d5ff <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf8f>
			4d5fb: R_X86_64_PC32	.rodata+0x78464
   4d5ff:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4d606:	00 
   4d607:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d60e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf9e>
			4d60a: R_X86_64_PC32	.rodata+0x78454
   4d60e:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4d615:	00 
   4d616:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4d61d:	00 
   4d61e:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4d625:	00 
   4d626:	4c 89 f7             	mov    %r14,%rdi
   4d629:	e8 00 00 00 00       	call   4d62e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xfbe>
			4d62a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d62e:	4c 89 f7             	mov    %r14,%rdi
   4d631:	e8 00 00 00 00       	call   4d636 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xfc6>
			4d632: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d636:	49 89 c6             	mov    %rax,%r14
   4d639:	4d 01 ff             	add    %r15,%r15
   4d63c:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4d643:	00 00 00 
   4d646:	4c 21 f8             	and    %r15,%rax
   4d649:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4d650:	00 01 00 00 00 
   4d655:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4d65c:	00 04 00 00 00 
   4d661:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4d668:	00 
   4d669:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4d670:	00 01 00 00 00 
   4d675:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4d67c:	00 04 00 00 00 
   4d681:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4d688:	00 20 00 00 00 
   4d68d:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d690:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4d697:	00 
   4d698:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4d69f:	00 00 00 00 00 
   4d6a4:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4d6ab:	00 
   4d6ac:	4c 89 ff             	mov    %r15,%rdi
   4d6af:	e8 00 00 00 00       	call   4d6b4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1044>
			4d6b0: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d6b4:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4d6bb:	00 
   4d6bc:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4d6c3:	00 
   4d6c4:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4d6cb:	00 
   4d6cc:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4d6d3:	00 
   4d6d4:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4d6db:	00 
   4d6dc:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4d6e3:	00 03 00 00 00 
   4d6e8:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d6ed:	4c 89 f6             	mov    %r14,%rsi
   4d6f0:	e8 00 00 00 00       	call   4d6f5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1085>
			4d6f1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d6f5:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d6fc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x108c>
			4d6f8: R_X86_64_PC32	.bss+0x1fb4
   4d6fc:	84 c0                	test   %al,%al
   4d6fe:	0f 84 ac 1c 00 00    	je     4f3b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d40>
   4d704:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d70b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x109b>
			4d707: R_X86_64_PC32	.bss+0x1fac
   4d70b:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4d712:	00 
   4d713:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4d71a:	00 
   4d71b:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4d722:	00 
   4d723:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4d72a:	00 
   4d72b:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4d732:	00 
   4d733:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4d73a:	00 
   4d73b:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4d742:	00 
   4d743:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4d74a:	00 
   4d74b:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4d752:	00 
   4d753:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4d75a:	00 
   4d75b:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4d762:	00 
   4d763:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4d76a:	00 
   4d76b:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4d772:	01 
   4d773:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4d77a:	00 00 00 00 00 
			4d77b: R_X86_64_32S	.rodata+0x49078
   4d77f:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d784:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4d78b:	00 
   4d78c:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d78f:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4d796:	00 
   4d797:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d79c:	e8 00 00 00 00       	call   4d7a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1131>
			4d79d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d7a1:	31 c9                	xor    %ecx,%ecx
   4d7a3:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d7a8:	0f 94 c1             	sete   %cl
   4d7ab:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4d7b2:	00 
   4d7b3:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d7b7:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d7bb:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d7c0:	48 8d bc 24 c8 03 00 	lea    0x3c8(%rsp),%rdi
   4d7c7:	00 
   4d7c8:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4d7cf:	00 
   4d7d0:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4d7d7:	00 
   4d7d8:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4d7df:	00 
   4d7e0:	48 89 de             	mov    %rbx,%rsi
   4d7e3:	4d 89 f0             	mov    %r14,%r8
   4d7e6:	e8 00 00 00 00       	call   4d7eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x117b>
			4d7e7: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d7eb:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d7f0:	e8 00 00 00 00       	call   4d7f5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1185>
			4d7f1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d7f5:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4d7fc:	00 
   4d7fd:	e8 00 00 00 00       	call   4d802 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1192>
			4d7fe: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d802:	48 8b 9c 24 d0 03 00 	mov    0x3d0(%rsp),%rbx
   4d809:	00 
   4d80a:	48 85 db             	test   %rbx,%rbx
   4d80d:	0f 85 bc 12 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4d813:	e9 12 13 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4d818:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d81d:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d822:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d827:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d82e:	44 89 c5             	mov    %r8d,%ebp
   4d831:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4d838:	00 
   4d839:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d840 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11d0>
			4d83c: R_X86_64_PC32	.rodata+0x78464
   4d840:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4d847:	00 
   4d848:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d84f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11df>
			4d84b: R_X86_64_PC32	.rodata+0x78454
   4d84f:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4d856:	00 
   4d857:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4d85e:	00 
   4d85f:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4d866:	00 
   4d867:	4c 89 f7             	mov    %r14,%rdi
   4d86a:	e8 00 00 00 00       	call   4d86f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11ff>
			4d86b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d86f:	4c 89 f7             	mov    %r14,%rdi
   4d872:	e8 00 00 00 00       	call   4d877 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1207>
			4d873: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d877:	49 89 c6             	mov    %rax,%r14
   4d87a:	48 63 c5             	movslq %ebp,%rax
   4d87d:	48 8d 04 c5 18 00 00 	lea    0x18(,%rax,8),%rax
   4d884:	00 
   4d885:	48 b9 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rcx
   4d88c:	00 00 00 
   4d88f:	48 21 c1             	and    %rax,%rcx
   4d892:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4d899:	00 01 00 00 00 
   4d89e:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4d8a5:	00 04 00 00 00 
   4d8aa:	48 89 8c 24 d8 00 00 	mov    %rcx,0xd8(%rsp)
   4d8b1:	00 
   4d8b2:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4d8b9:	00 01 00 00 00 
   4d8be:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4d8c5:	00 04 00 00 00 
   4d8ca:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4d8d1:	00 20 00 00 00 
   4d8d6:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d8d9:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4d8e0:	00 
   4d8e1:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4d8e8:	00 00 00 00 00 
   4d8ed:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4d8f4:	00 
   4d8f5:	4c 89 ff             	mov    %r15,%rdi
   4d8f8:	e8 00 00 00 00       	call   4d8fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x128d>
			4d8f9: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d8fd:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4d904:	00 
   4d905:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4d90c:	00 
   4d90d:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4d914:	00 
   4d915:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4d91c:	00 
   4d91d:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4d924:	00 
   4d925:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4d92c:	00 03 00 00 00 
   4d931:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d936:	4c 89 f6             	mov    %r14,%rsi
   4d939:	e8 00 00 00 00       	call   4d93e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12ce>
			4d93a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d93e:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d945 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12d5>
			4d941: R_X86_64_PC32	.bss+0x1fd4
   4d945:	84 c0                	test   %al,%al
   4d947:	0f 84 dc 18 00 00    	je     4f229 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bb9>
   4d94d:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d954 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12e4>
			4d950: R_X86_64_PC32	.bss+0x1fcc
   4d954:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4d95b:	00 
   4d95c:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4d963:	00 
   4d964:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4d96b:	00 
   4d96c:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4d973:	00 
   4d974:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4d97b:	00 
   4d97c:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4d983:	00 
   4d984:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4d98b:	00 
   4d98c:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4d993:	00 
   4d994:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4d99b:	00 
   4d99c:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4d9a3:	00 
   4d9a4:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4d9ab:	00 
   4d9ac:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4d9b3:	00 
   4d9b4:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4d9bb:	01 
   4d9bc:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4d9c3:	00 00 00 00 00 
			4d9c4: R_X86_64_32S	.rodata+0x49488
   4d9c8:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d9cd:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4d9d4:	00 
   4d9d5:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d9d8:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4d9df:	00 
   4d9e0:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d9e5:	e8 00 00 00 00       	call   4d9ea <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x137a>
			4d9e6: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d9ea:	31 c9                	xor    %ecx,%ecx
   4d9ec:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d9f1:	0f 94 c1             	sete   %cl
   4d9f4:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4d9fb:	00 
   4d9fc:	0f 11 04 24          	movups %xmm0,(%rsp)
   4da00:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4da04:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4da09:	48 8d bc 24 a8 03 00 	lea    0x3a8(%rsp),%rdi
   4da10:	00 
   4da11:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4da18:	00 
   4da19:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4da20:	00 
   4da21:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4da28:	00 
   4da29:	48 89 de             	mov    %rbx,%rsi
   4da2c:	4d 89 f0             	mov    %r14,%r8
   4da2f:	e8 00 00 00 00       	call   4da34 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13c4>
			4da30: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4da34:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4da39:	e8 00 00 00 00       	call   4da3e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13ce>
			4da3a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4da3e:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4da45:	00 
   4da46:	e8 00 00 00 00       	call   4da4b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13db>
			4da47: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4da4b:	48 8b 9c 24 b0 03 00 	mov    0x3b0(%rsp),%rbx
   4da52:	00 
   4da53:	48 85 db             	test   %rbx,%rbx
   4da56:	0f 85 73 10 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4da5c:	e9 c9 10 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4da61:	0f 57 c0             	xorps  %xmm0,%xmm0
   4da64:	0f 29 84 24 f0 02 00 	movaps %xmm0,0x2f0(%rsp)
   4da6b:	00 
   4da6c:	48 c7 84 24 00 03 00 	movq   $0x0,0x300(%rsp)
   4da73:	00 00 00 00 00 
   4da78:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4da7d:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4da82:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4da87:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4da8e:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4da95:	00 
   4da96:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4da9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x142d>
			4da99: R_X86_64_PC32	.rodata+0x78464
   4da9d:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4daa4:	00 
   4daa5:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4daac <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x143c>
			4daa8: R_X86_64_PC32	.rodata+0x78454
   4daac:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4dab3:	00 
   4dab4:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4dabb:	00 
   4dabc:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4dac3:	00 
   4dac4:	4c 89 f7             	mov    %r14,%rdi
   4dac7:	e8 00 00 00 00       	call   4dacc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x145c>
			4dac8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4dacc:	4c 89 f7             	mov    %r14,%rdi
   4dacf:	e8 00 00 00 00       	call   4dad4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1464>
			4dad0: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dad4:	49 89 c6             	mov    %rax,%r14
   4dad7:	49 c1 e7 03          	shl    $0x3,%r15
   4dadb:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4dae2:	00 00 00 
   4dae5:	4c 21 f8             	and    %r15,%rax
   4dae8:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4daef:	00 01 00 00 00 
   4daf4:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4dafb:	00 04 00 00 00 
   4db00:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4db07:	00 
   4db08:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4db0f:	00 01 00 00 00 
   4db14:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4db1b:	00 04 00 00 00 
   4db20:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4db27:	00 20 00 00 00 
   4db2c:	0f 28 84 24 f0 02 00 	movaps 0x2f0(%rsp),%xmm0
   4db33:	00 
   4db34:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4db3b:	00 
   4db3c:	48 8b 84 24 00 03 00 	mov    0x300(%rsp),%rax
   4db43:	00 
   4db44:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4db4b:	00 
   4db4c:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4db53:	00 
   4db54:	4c 89 ff             	mov    %r15,%rdi
   4db57:	e8 00 00 00 00       	call   4db5c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x14ec>
			4db58: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4db5c:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4db63:	00 
   4db64:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4db6b:	00 
   4db6c:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4db73:	00 
   4db74:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4db7b:	00 
   4db7c:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4db83:	00 
   4db84:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4db8b:	00 03 00 00 00 
   4db90:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4db95:	4c 89 f6             	mov    %r14,%rsi
   4db98:	e8 00 00 00 00       	call   4db9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x152d>
			4db99: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4db9d:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4dba4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1534>
			4dba0: R_X86_64_PC32	.bss+0x1f04
   4dba4:	84 c0                	test   %al,%al
   4dba6:	0f 84 bf 18 00 00    	je     4f46b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dfb>
   4dbac:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4dbb3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1543>
			4dbaf: R_X86_64_PC32	.bss+0x1efc
   4dbb3:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4dbba:	00 
   4dbbb:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4dbc2:	00 
   4dbc3:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4dbca:	00 
   4dbcb:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4dbd2:	00 
   4dbd3:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4dbda:	00 
   4dbdb:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4dbe2:	00 
   4dbe3:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4dbea:	00 
   4dbeb:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4dbf2:	00 
   4dbf3:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4dbfa:	00 
   4dbfb:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4dc02:	00 
   4dc03:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4dc0a:	00 
   4dc0b:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4dc12:	00 
   4dc13:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4dc1a:	01 
   4dc1b:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4dc22:	00 00 00 00 00 
			4dc23: R_X86_64_32S	.rodata+0x47c28
   4dc27:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4dc2c:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4dc33:	00 
   4dc34:	0f 57 c0             	xorps  %xmm0,%xmm0
   4dc37:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4dc3e:	00 
   4dc3f:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dc44:	e8 00 00 00 00       	call   4dc49 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x15d9>
			4dc45: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dc49:	31 c9                	xor    %ecx,%ecx
   4dc4b:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4dc50:	0f 94 c1             	sete   %cl
   4dc53:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4dc5a:	00 
   4dc5b:	0f 11 04 24          	movups %xmm0,(%rsp)
   4dc5f:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4dc63:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4dc68:	48 8d bc 24 38 04 00 	lea    0x438(%rsp),%rdi
   4dc6f:	00 
   4dc70:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4dc77:	00 
   4dc78:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4dc7f:	00 
   4dc80:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4dc87:	00 
   4dc88:	48 89 de             	mov    %rbx,%rsi
   4dc8b:	4d 89 f0             	mov    %r14,%r8
   4dc8e:	e8 00 00 00 00       	call   4dc93 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1623>
			4dc8f: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4dc93:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dc98:	e8 00 00 00 00       	call   4dc9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x162d>
			4dc99: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4dc9d:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4dca4:	00 
   4dca5:	e8 00 00 00 00       	call   4dcaa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x163a>
			4dca6: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4dcaa:	48 8b 9c 24 40 04 00 	mov    0x440(%rsp),%rbx
   4dcb1:	00 
   4dcb2:	48 85 db             	test   %rbx,%rbx
   4dcb5:	0f 85 14 0e 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4dcbb:	e9 6a 0e 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4dcc0:	0f 57 c0             	xorps  %xmm0,%xmm0
   4dcc3:	0f 29 84 24 70 03 00 	movaps %xmm0,0x370(%rsp)
   4dcca:	00 
   4dccb:	48 c7 84 24 80 03 00 	movq   $0x0,0x380(%rsp)
   4dcd2:	00 00 00 00 00 
   4dcd7:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4dcdc:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4dce1:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4dce6:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4dced:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4dcf4:	00 
   4dcf5:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4dcfc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x168c>
			4dcf8: R_X86_64_PC32	.rodata+0x78464
   4dcfc:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4dd03:	00 
   4dd04:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4dd0b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x169b>
			4dd07: R_X86_64_PC32	.rodata+0x78454
   4dd0b:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4dd12:	00 
   4dd13:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4dd1a:	00 
   4dd1b:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4dd22:	00 
   4dd23:	4c 89 f7             	mov    %r14,%rdi
   4dd26:	e8 00 00 00 00       	call   4dd2b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x16bb>
			4dd27: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4dd2b:	4c 89 f7             	mov    %r14,%rdi
   4dd2e:	e8 00 00 00 00       	call   4dd33 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x16c3>
			4dd2f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dd33:	49 89 c6             	mov    %rax,%r14
   4dd36:	49 c1 e7 03          	shl    $0x3,%r15
   4dd3a:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4dd41:	00 00 00 
   4dd44:	4c 21 f8             	and    %r15,%rax
   4dd47:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4dd4e:	00 01 00 00 00 
   4dd53:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4dd5a:	00 04 00 00 00 
   4dd5f:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4dd66:	00 
   4dd67:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4dd6e:	00 01 00 00 00 
   4dd73:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4dd7a:	00 04 00 00 00 
   4dd7f:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4dd86:	00 20 00 00 00 
   4dd8b:	0f 28 84 24 70 03 00 	movaps 0x370(%rsp),%xmm0
   4dd92:	00 
   4dd93:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4dd9a:	00 
   4dd9b:	48 8b 84 24 80 03 00 	mov    0x380(%rsp),%rax
   4dda2:	00 
   4dda3:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4ddaa:	00 
   4ddab:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4ddb2:	00 
   4ddb3:	4c 89 ff             	mov    %r15,%rdi
   4ddb6:	e8 00 00 00 00       	call   4ddbb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x174b>
			4ddb7: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4ddbb:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4ddc2:	00 
   4ddc3:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4ddca:	00 
   4ddcb:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4ddd2:	00 
   4ddd3:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4ddda:	00 
   4dddb:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4dde2:	00 
   4dde3:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4ddea:	00 03 00 00 00 
   4ddef:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ddf4:	4c 89 f6             	mov    %r14,%rsi
   4ddf7:	e8 00 00 00 00       	call   4ddfc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x178c>
			4ddf8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4ddfc:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4de03 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1793>
			4ddff: R_X86_64_PC32	.bss+0x1ec4
   4de03:	84 c0                	test   %al,%al
   4de05:	0f 84 92 16 00 00    	je     4f49d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e2d>
   4de0b:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4de12 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x17a2>
			4de0e: R_X86_64_PC32	.bss+0x1ebc
   4de12:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4de19:	00 
   4de1a:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4de21:	00 
   4de22:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4de29:	00 
   4de2a:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4de31:	00 
   4de32:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4de39:	00 
   4de3a:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4de41:	00 
   4de42:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4de49:	00 
   4de4a:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4de51:	00 
   4de52:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4de59:	00 
   4de5a:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4de61:	00 
   4de62:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4de69:	00 
   4de6a:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4de71:	00 
   4de72:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4de79:	01 
   4de7a:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4de81:	00 00 00 00 00 
			4de82: R_X86_64_32S	.rodata+0x47408
   4de86:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4de8b:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4de92:	00 
   4de93:	0f 57 c0             	xorps  %xmm0,%xmm0
   4de96:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4de9d:	00 
   4de9e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dea3:	e8 00 00 00 00       	call   4dea8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1838>
			4dea4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dea8:	31 c9                	xor    %ecx,%ecx
   4deaa:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4deaf:	0f 94 c1             	sete   %cl
   4deb2:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4deb9:	00 
   4deba:	0f 11 04 24          	movups %xmm0,(%rsp)
   4debe:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4dec2:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4dec7:	48 8d bc 24 78 04 00 	lea    0x478(%rsp),%rdi
   4dece:	00 
   4decf:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4ded6:	00 
   4ded7:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4dede:	00 
   4dedf:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4dee6:	00 
   4dee7:	48 89 de             	mov    %rbx,%rsi
   4deea:	4d 89 f0             	mov    %r14,%r8
   4deed:	e8 00 00 00 00       	call   4def2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1882>
			4deee: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4def2:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4def7:	e8 00 00 00 00       	call   4defc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x188c>
			4def8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4defc:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4df03:	00 
   4df04:	e8 00 00 00 00       	call   4df09 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1899>
			4df05: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4df09:	48 8b 9c 24 80 04 00 	mov    0x480(%rsp),%rbx
   4df10:	00 
   4df11:	48 85 db             	test   %rbx,%rbx
   4df14:	0f 85 b5 0b 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4df1a:	e9 0b 0c 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4df1f:	0f 57 c0             	xorps  %xmm0,%xmm0
   4df22:	0f 29 84 24 30 03 00 	movaps %xmm0,0x330(%rsp)
   4df29:	00 
   4df2a:	48 c7 84 24 40 03 00 	movq   $0x0,0x340(%rsp)
   4df31:	00 00 00 00 00 
   4df36:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4df3b:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4df40:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4df45:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4df4c:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4df53:	00 
   4df54:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4df5b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x18eb>
			4df57: R_X86_64_PC32	.rodata+0x78464
   4df5b:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4df62:	00 
   4df63:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4df6a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x18fa>
			4df66: R_X86_64_PC32	.rodata+0x78454
   4df6a:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4df71:	00 
   4df72:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4df79:	00 
   4df7a:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4df81:	00 
   4df82:	4c 89 f7             	mov    %r14,%rdi
   4df85:	e8 00 00 00 00       	call   4df8a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x191a>
			4df86: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4df8a:	4c 89 f7             	mov    %r14,%rdi
   4df8d:	e8 00 00 00 00       	call   4df92 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1922>
			4df8e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4df92:	49 89 c6             	mov    %rax,%r14
   4df95:	49 c1 e7 03          	shl    $0x3,%r15
   4df99:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4dfa0:	00 00 00 
   4dfa3:	4c 21 f8             	and    %r15,%rax
   4dfa6:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4dfad:	00 01 00 00 00 
   4dfb2:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4dfb9:	00 04 00 00 00 
   4dfbe:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4dfc5:	00 
   4dfc6:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4dfcd:	00 01 00 00 00 
   4dfd2:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4dfd9:	00 04 00 00 00 
   4dfde:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4dfe5:	00 20 00 00 00 
   4dfea:	0f 28 84 24 30 03 00 	movaps 0x330(%rsp),%xmm0
   4dff1:	00 
   4dff2:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4dff9:	00 
   4dffa:	48 8b 84 24 40 03 00 	mov    0x340(%rsp),%rax
   4e001:	00 
   4e002:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4e009:	00 
   4e00a:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4e011:	00 
   4e012:	4c 89 ff             	mov    %r15,%rdi
   4e015:	e8 00 00 00 00       	call   4e01a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19aa>
			4e016: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e01a:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4e021:	00 
   4e022:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4e029:	00 
   4e02a:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4e031:	00 
   4e032:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4e039:	00 
   4e03a:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4e041:	00 
   4e042:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4e049:	00 03 00 00 00 
   4e04e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e053:	4c 89 f6             	mov    %r14,%rsi
   4e056:	e8 00 00 00 00       	call   4e05b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19eb>
			4e057: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e05b:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e062 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19f2>
			4e05e: R_X86_64_PC32	.bss+0x1ee4
   4e062:	84 c0                	test   %al,%al
   4e064:	0f 84 65 14 00 00    	je     4f4cf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e5f>
   4e06a:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e071 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1a01>
			4e06d: R_X86_64_PC32	.bss+0x1edc
   4e071:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4e078:	00 
   4e079:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4e080:	00 
   4e081:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4e088:	00 
   4e089:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4e090:	00 
   4e091:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4e098:	00 
   4e099:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4e0a0:	00 
   4e0a1:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4e0a8:	00 
   4e0a9:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4e0b0:	00 
   4e0b1:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4e0b8:	00 
   4e0b9:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4e0c0:	00 
   4e0c1:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4e0c8:	00 
   4e0c9:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4e0d0:	00 
   4e0d1:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4e0d8:	01 
   4e0d9:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4e0e0:	00 00 00 00 00 
			4e0e1: R_X86_64_32S	.rodata+0x47818
   4e0e5:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e0ea:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4e0f1:	00 
   4e0f2:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e0f5:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4e0fc:	00 
   4e0fd:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e102:	e8 00 00 00 00       	call   4e107 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1a97>
			4e103: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e107:	31 c9                	xor    %ecx,%ecx
   4e109:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e10e:	0f 94 c1             	sete   %cl
   4e111:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4e118:	00 
   4e119:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e11d:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e121:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e126:	48 8d bc 24 58 04 00 	lea    0x458(%rsp),%rdi
   4e12d:	00 
   4e12e:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4e135:	00 
   4e136:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4e13d:	00 
   4e13e:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4e145:	00 
   4e146:	48 89 de             	mov    %rbx,%rsi
   4e149:	4d 89 f0             	mov    %r14,%r8
   4e14c:	e8 00 00 00 00       	call   4e151 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1ae1>
			4e14d: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e151:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e156:	e8 00 00 00 00       	call   4e15b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1aeb>
			4e157: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e15b:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4e162:	00 
   4e163:	e8 00 00 00 00       	call   4e168 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1af8>
			4e164: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e168:	48 8b 9c 24 60 04 00 	mov    0x460(%rsp),%rbx
   4e16f:	00 
   4e170:	48 85 db             	test   %rbx,%rbx
   4e173:	0f 85 56 09 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4e179:	e9 ac 09 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4e17e:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e181:	0f 29 84 24 b0 02 00 	movaps %xmm0,0x2b0(%rsp)
   4e188:	00 
   4e189:	48 c7 84 24 c0 02 00 	movq   $0x0,0x2c0(%rsp)
   4e190:	00 00 00 00 00 
   4e195:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e19a:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e19f:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e1a4:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e1ab:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4e1b2:	00 
   4e1b3:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e1ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b4a>
			4e1b6: R_X86_64_PC32	.rodata+0x78464
   4e1ba:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4e1c1:	00 
   4e1c2:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e1c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b59>
			4e1c5: R_X86_64_PC32	.rodata+0x78454
   4e1c9:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4e1d0:	00 
   4e1d1:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4e1d8:	00 
   4e1d9:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4e1e0:	00 
   4e1e1:	4c 89 f7             	mov    %r14,%rdi
   4e1e4:	e8 00 00 00 00       	call   4e1e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b79>
			4e1e5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e1e9:	4c 89 f7             	mov    %r14,%rdi
   4e1ec:	e8 00 00 00 00       	call   4e1f1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b81>
			4e1ed: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e1f1:	49 89 c6             	mov    %rax,%r14
   4e1f4:	49 c1 e7 03          	shl    $0x3,%r15
   4e1f8:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4e1ff:	00 00 00 
   4e202:	4c 21 f8             	and    %r15,%rax
   4e205:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4e20c:	00 01 00 00 00 
   4e211:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4e218:	00 04 00 00 00 
   4e21d:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4e224:	00 
   4e225:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4e22c:	00 01 00 00 00 
   4e231:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4e238:	00 04 00 00 00 
   4e23d:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4e244:	00 20 00 00 00 
   4e249:	0f 28 84 24 b0 02 00 	movaps 0x2b0(%rsp),%xmm0
   4e250:	00 
   4e251:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4e258:	00 
   4e259:	48 8b 84 24 c0 02 00 	mov    0x2c0(%rsp),%rax
   4e260:	00 
   4e261:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
   4e268:	00 
   4e269:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4e270:	00 
   4e271:	4c 89 ff             	mov    %r15,%rdi
   4e274:	e8 00 00 00 00       	call   4e279 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c09>
			4e275: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e279:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4e280:	00 
   4e281:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4e288:	00 
   4e289:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4e290:	00 
   4e291:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4e298:	00 
   4e299:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4e2a0:	00 
   4e2a1:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4e2a8:	00 03 00 00 00 
   4e2ad:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e2b2:	4c 89 f6             	mov    %r14,%rsi
   4e2b5:	e8 00 00 00 00       	call   4e2ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c4a>
			4e2b6: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e2ba:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e2c1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c51>
			4e2bd: R_X86_64_PC32	.bss+0x1f24
   4e2c1:	84 c0                	test   %al,%al
   4e2c3:	0f 84 38 12 00 00    	je     4f501 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e91>
   4e2c9:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e2d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c60>
			4e2cc: R_X86_64_PC32	.bss+0x1f1c
   4e2d0:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4e2d7:	00 
   4e2d8:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4e2df:	00 
   4e2e0:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4e2e7:	00 
   4e2e8:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4e2ef:	00 
   4e2f0:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4e2f7:	00 
   4e2f8:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4e2ff:	00 
   4e300:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4e307:	00 
   4e308:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4e30f:	00 
   4e310:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4e317:	00 
   4e318:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4e31f:	00 
   4e320:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4e327:	00 
   4e328:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4e32f:	00 
   4e330:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4e337:	01 
   4e338:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4e33f:	00 00 00 00 00 
			4e340: R_X86_64_32S	.rodata+0x48038
   4e344:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e349:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4e350:	00 
   4e351:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e354:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4e35b:	00 
   4e35c:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e361:	e8 00 00 00 00       	call   4e366 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1cf6>
			4e362: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e366:	31 c9                	xor    %ecx,%ecx
   4e368:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e36d:	0f 94 c1             	sete   %cl
   4e370:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4e377:	00 
   4e378:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e37c:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e380:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e385:	48 8d bc 24 18 04 00 	lea    0x418(%rsp),%rdi
   4e38c:	00 
   4e38d:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4e394:	00 
   4e395:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4e39c:	00 
   4e39d:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4e3a4:	00 
   4e3a5:	48 89 de             	mov    %rbx,%rsi
   4e3a8:	4d 89 f0             	mov    %r14,%r8
   4e3ab:	e8 00 00 00 00       	call   4e3b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d40>
			4e3ac: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e3b0:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e3b5:	e8 00 00 00 00       	call   4e3ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d4a>
			4e3b6: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e3ba:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4e3c1:	00 
   4e3c2:	e8 00 00 00 00       	call   4e3c7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d57>
			4e3c3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e3c7:	48 8b 9c 24 20 04 00 	mov    0x420(%rsp),%rbx
   4e3ce:	00 
   4e3cf:	48 85 db             	test   %rbx,%rbx
   4e3d2:	0f 85 f7 06 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4e3d8:	e9 4d 07 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4e3dd:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e3e2:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e3e7:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e3ec:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e3f3:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4e3fa:	00 
   4e3fb:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e402 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d92>
			4e3fe: R_X86_64_PC32	.rodata+0x78464
   4e402:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4e409:	00 
   4e40a:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e411 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1da1>
			4e40d: R_X86_64_PC32	.rodata+0x78454
   4e411:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4e418:	00 
   4e419:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4e420:	00 
   4e421:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4e428:	00 
   4e429:	4c 89 f7             	mov    %r14,%rdi
   4e42c:	e8 00 00 00 00       	call   4e431 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1dc1>
			4e42d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e431:	4c 89 f7             	mov    %r14,%rdi
   4e434:	e8 00 00 00 00       	call   4e439 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1dc9>
			4e435: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e439:	49 89 c6             	mov    %rax,%r14
   4e43c:	49 c1 e7 02          	shl    $0x2,%r15
   4e440:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4e447:	00 00 00 
   4e44a:	4c 21 f8             	and    %r15,%rax
   4e44d:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4e454:	00 01 00 00 00 
   4e459:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4e460:	00 04 00 00 00 
   4e465:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4e46c:	00 
   4e46d:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4e474:	00 01 00 00 00 
   4e479:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4e480:	00 04 00 00 00 
   4e485:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4e48c:	00 20 00 00 00 
   4e491:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e494:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4e49b:	00 
   4e49c:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4e4a3:	00 00 00 00 00 
   4e4a8:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4e4af:	00 
   4e4b0:	4c 89 ff             	mov    %r15,%rdi
   4e4b3:	e8 00 00 00 00       	call   4e4b8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e48>
			4e4b4: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e4b8:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4e4bf:	00 
   4e4c0:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4e4c7:	00 
   4e4c8:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4e4cf:	00 
   4e4d0:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4e4d7:	00 
   4e4d8:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4e4df:	00 
   4e4e0:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4e4e7:	00 03 00 00 00 
   4e4ec:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e4f1:	4c 89 f6             	mov    %r14,%rsi
   4e4f4:	e8 00 00 00 00       	call   4e4f9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e89>
			4e4f5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e4f9:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e500 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e90>
			4e4fc: R_X86_64_PC32	.bss+0x1fa4
   4e500:	84 c0                	test   %al,%al
   4e502:	0f 84 2b 10 00 00    	je     4f533 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ec3>
   4e508:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e50f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e9f>
			4e50b: R_X86_64_PC32	.bss+0x1f9c
   4e50f:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4e516:	00 
   4e517:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4e51e:	00 
   4e51f:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4e526:	00 
   4e527:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4e52e:	00 
   4e52f:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4e536:	00 
   4e537:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4e53e:	00 
   4e53f:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4e546:	00 
   4e547:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4e54e:	00 
   4e54f:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4e556:	00 
   4e557:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4e55e:	00 
   4e55f:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4e566:	00 
   4e567:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4e56e:	00 
   4e56f:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4e576:	01 
   4e577:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4e57e:	00 00 00 00 00 
			4e57f: R_X86_64_32S	.rodata+0x48e70
   4e583:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e588:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4e58f:	00 
   4e590:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e593:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4e59a:	00 
   4e59b:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e5a0:	e8 00 00 00 00       	call   4e5a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f35>
			4e5a1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e5a5:	31 c9                	xor    %ecx,%ecx
   4e5a7:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e5ac:	0f 94 c1             	sete   %cl
   4e5af:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4e5b6:	00 
   4e5b7:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e5bb:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e5bf:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e5c4:	48 8d bc 24 d8 03 00 	lea    0x3d8(%rsp),%rdi
   4e5cb:	00 
   4e5cc:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4e5d3:	00 
   4e5d4:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4e5db:	00 
   4e5dc:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4e5e3:	00 
   4e5e4:	48 89 de             	mov    %rbx,%rsi
   4e5e7:	4d 89 f0             	mov    %r14,%r8
   4e5ea:	e8 00 00 00 00       	call   4e5ef <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f7f>
			4e5eb: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e5ef:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e5f4:	e8 00 00 00 00       	call   4e5f9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f89>
			4e5f5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e5f9:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4e600:	00 
   4e601:	e8 00 00 00 00       	call   4e606 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f96>
			4e602: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e606:	48 8b 9c 24 e0 03 00 	mov    0x3e0(%rsp),%rbx
   4e60d:	00 
   4e60e:	48 85 db             	test   %rbx,%rbx
   4e611:	0f 85 b8 04 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4e617:	e9 0e 05 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4e61c:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e621:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e626:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e62b:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e632:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4e639:	00 
   4e63a:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e641 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fd1>
			4e63d: R_X86_64_PC32	.rodata+0x78464
   4e641:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4e648:	00 
   4e649:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e650 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fe0>
			4e64c: R_X86_64_PC32	.rodata+0x78454
   4e650:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4e657:	00 
   4e658:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4e65f:	00 
   4e660:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4e667:	00 
   4e668:	4c 89 f7             	mov    %r14,%rdi
   4e66b:	e8 00 00 00 00       	call   4e670 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2000>
			4e66c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e670:	4c 89 f7             	mov    %r14,%rdi
   4e673:	e8 00 00 00 00       	call   4e678 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2008>
			4e674: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e678:	49 89 c6             	mov    %rax,%r14
   4e67b:	4d 01 ff             	add    %r15,%r15
   4e67e:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4e685:	00 00 00 
   4e688:	4c 21 f8             	and    %r15,%rax
   4e68b:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4e692:	00 01 00 00 00 
   4e697:	48 c7 84 24 d0 00 00 	movq   $0x4,0xd0(%rsp)
   4e69e:	00 04 00 00 00 
   4e6a3:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   4e6aa:	00 
   4e6ab:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4e6b2:	00 01 00 00 00 
   4e6b7:	48 c7 84 24 e8 00 00 	movq   $0x4,0xe8(%rsp)
   4e6be:	00 04 00 00 00 
   4e6c3:	48 c7 84 24 f0 00 00 	movq   $0x20,0xf0(%rsp)
   4e6ca:	00 20 00 00 00 
   4e6cf:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e6d2:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4e6d9:	00 
   4e6da:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4e6e1:	00 00 00 00 00 
   4e6e6:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4e6ed:	00 
   4e6ee:	4c 89 ff             	mov    %r15,%rdi
   4e6f1:	e8 00 00 00 00       	call   4e6f6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2086>
			4e6f2: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e6f6:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4e6fd:	00 
   4e6fe:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4e705:	00 
   4e706:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4e70d:	00 
   4e70e:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4e715:	00 
   4e716:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4e71d:	00 
   4e71e:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4e725:	00 03 00 00 00 
   4e72a:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e72f:	4c 89 f6             	mov    %r14,%rsi
   4e732:	e8 00 00 00 00       	call   4e737 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20c7>
			4e733: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e737:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e73e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20ce>
			4e73a: R_X86_64_PC32	.bss+0x1fc4
   4e73e:	84 c0                	test   %al,%al
   4e740:	0f 84 1f 0e 00 00    	je     4f565 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ef5>
   4e746:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e74d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20dd>
			4e749: R_X86_64_PC32	.bss+0x1fbc
   4e74d:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4e754:	00 
   4e755:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4e75c:	00 
   4e75d:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4e764:	00 
   4e765:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4e76c:	00 
   4e76d:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4e774:	00 
   4e775:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4e77c:	00 
   4e77d:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4e784:	00 
   4e785:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4e78c:	00 
   4e78d:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4e794:	00 
   4e795:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4e79c:	00 
   4e79d:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4e7a4:	00 
   4e7a5:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4e7ac:	00 
   4e7ad:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4e7b4:	01 
   4e7b5:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4e7bc:	00 00 00 00 00 
			4e7bd: R_X86_64_32S	.rodata+0x49280
   4e7c1:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e7c6:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4e7cd:	00 
   4e7ce:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e7d1:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4e7d8:	00 
   4e7d9:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e7de:	e8 00 00 00 00       	call   4e7e3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2173>
			4e7df: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e7e3:	31 c9                	xor    %ecx,%ecx
   4e7e5:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e7ea:	0f 94 c1             	sete   %cl
   4e7ed:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4e7f4:	00 
   4e7f5:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e7f9:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e7fd:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e802:	48 8d bc 24 b8 03 00 	lea    0x3b8(%rsp),%rdi
   4e809:	00 
   4e80a:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4e811:	00 
   4e812:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4e819:	00 
   4e81a:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4e821:	00 
   4e822:	48 89 de             	mov    %rbx,%rsi
   4e825:	4d 89 f0             	mov    %r14,%r8
   4e828:	e8 00 00 00 00       	call   4e82d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21bd>
			4e829: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e82d:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e832:	e8 00 00 00 00       	call   4e837 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21c7>
			4e833: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e837:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4e83e:	00 
   4e83f:	e8 00 00 00 00       	call   4e844 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21d4>
			4e840: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e844:	48 8b 9c 24 c0 03 00 	mov    0x3c0(%rsp),%rbx
   4e84b:	00 
   4e84c:	48 85 db             	test   %rbx,%rbx
   4e84f:	0f 85 7a 02 00 00    	jne    4eacf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x245f>
   4e855:	e9 d0 02 00 00       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4e85a:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e861 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21f1>
			4e85d: R_X86_64_PC32	.bss+0x4d4
   4e861:	84 c0                	test   %al,%al
   4e863:	0f 84 ab 0b 00 00    	je     4f414 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2da4>
   4e869:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 4e86f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21ff>
			4e86b: R_X86_64_PC32	.bss+0x4cc
   4e86f:	4d 63 f8             	movslq %r8d,%r15
   4e872:	83 f8 10             	cmp    $0x10,%eax
   4e875:	74 1e                	je     4e895 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2225>
   4e877:	a8 02                	test   $0x2,%al
   4e879:	0f 84 0f 04 00 00    	je     4ec8e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x261e>
   4e87f:	41 81 f8 00 04 00 00 	cmp    $0x400,%r8d
   4e886:	0f 8f 02 04 00 00    	jg     4ec8e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x261e>
   4e88c:	83 f8 04             	cmp    $0x4,%eax
   4e88f:	0f 8d f9 03 00 00    	jge    4ec8e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x261e>
   4e895:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e89a:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e89f:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e8a4:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e8ab:	44 89 84 24 84 00 00 	mov    %r8d,0x84(%rsp)
   4e8b2:	00 
   4e8b3:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e8ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x224a>
			4e8b6: R_X86_64_PC32	.rodata+0x78464
   4e8ba:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
   4e8c1:	00 
   4e8c2:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e8c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2259>
			4e8c5: R_X86_64_PC32	.rodata+0x78454
   4e8c9:	0f 29 84 24 10 01 00 	movaps %xmm0,0x110(%rsp)
   4e8d0:	00 
   4e8d1:	4c 8d b4 24 88 00 00 	lea    0x88(%rsp),%r14
   4e8d8:	00 
   4e8d9:	48 8d b4 24 10 01 00 	lea    0x110(%rsp),%rsi
   4e8e0:	00 
   4e8e1:	4c 89 f7             	mov    %r14,%rdi
   4e8e4:	e8 00 00 00 00       	call   4e8e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2279>
			4e8e5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e8e9:	4c 89 f7             	mov    %r14,%rdi
   4e8ec:	e8 00 00 00 00       	call   4e8f1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2281>
			4e8ed: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e8f1:	49 89 c6             	mov    %rax,%r14
   4e8f4:	4a 8d 04 7d 0e 00 00 	lea    0xe(,%r15,2),%rax
   4e8fb:	00 
   4e8fc:	48 b9 f0 ff ff ff 0f 	movabs $0xffffffff0,%rcx
   4e903:	00 00 00 
   4e906:	48 21 c1             	and    %rax,%rcx
   4e909:	48 c7 84 24 c8 00 00 	movq   $0x1,0xc8(%rsp)
   4e910:	00 01 00 00 00 
   4e915:	48 c7 84 24 d0 00 00 	movq   $0x8,0xd0(%rsp)
   4e91c:	00 08 00 00 00 
   4e921:	48 89 8c 24 d8 00 00 	mov    %rcx,0xd8(%rsp)
   4e928:	00 
   4e929:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
   4e930:	00 01 00 00 00 
   4e935:	48 c7 84 24 e8 00 00 	movq   $0x8,0xe8(%rsp)
   4e93c:	00 08 00 00 00 
   4e941:	48 c7 84 24 f0 00 00 	movq   $0x10,0xf0(%rsp)
   4e948:	00 10 00 00 00 
   4e94d:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e950:	0f 11 84 24 f8 00 00 	movups %xmm0,0xf8(%rsp)
   4e957:	00 
   4e958:	48 c7 84 24 08 01 00 	movq   $0x0,0x108(%rsp)
   4e95f:	00 00 00 00 00 
   4e964:	4c 8d bc 24 c8 00 00 	lea    0xc8(%rsp),%r15
   4e96b:	00 
   4e96c:	4c 89 ff             	mov    %r15,%rdi
   4e96f:	e8 00 00 00 00       	call   4e974 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2304>
			4e970: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e974:	48 8d 84 24 e0 00 00 	lea    0xe0(%rsp),%rax
   4e97b:	00 
   4e97c:	48 8d 8c 24 f8 00 00 	lea    0xf8(%rsp),%rcx
   4e983:	00 
   4e984:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
   4e98b:	00 
   4e98c:	48 89 84 24 b0 00 00 	mov    %rax,0xb0(%rsp)
   4e993:	00 
   4e994:	48 89 8c 24 b8 00 00 	mov    %rcx,0xb8(%rsp)
   4e99b:	00 
   4e99c:	48 c7 84 24 c0 00 00 	movq   $0x3,0xc0(%rsp)
   4e9a3:	00 03 00 00 00 
   4e9a8:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e9ad:	4c 89 f6             	mov    %r14,%rsi
   4e9b0:	e8 00 00 00 00       	call   4e9b5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2345>
			4e9b1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e9b5:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e9bc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x234c>
			4e9b8: R_X86_64_PC32	.bss+0x1f64
   4e9bc:	84 c0                	test   %al,%al
   4e9be:	0f 84 d3 0b 00 00    	je     4f597 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f27>
   4e9c4:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e9cb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x235b>
			4e9c7: R_X86_64_PC32	.bss+0x1f5c
   4e9cb:	c6 84 24 48 01 00 00 	movb   $0x0,0x148(%rsp)
   4e9d2:	00 
   4e9d3:	c6 84 24 52 01 00 00 	movb   $0x0,0x152(%rsp)
   4e9da:	00 
   4e9db:	c6 84 24 54 01 00 00 	movb   $0x0,0x154(%rsp)
   4e9e2:	00 
   4e9e3:	c6 84 24 5c 01 00 00 	movb   $0x0,0x15c(%rsp)
   4e9ea:	00 
   4e9eb:	c6 84 24 64 01 00 00 	movb   $0x0,0x164(%rsp)
   4e9f2:	00 
   4e9f3:	c6 84 24 6c 01 00 00 	movb   $0x0,0x16c(%rsp)
   4e9fa:	00 
   4e9fb:	c6 84 24 74 01 00 00 	movb   $0x0,0x174(%rsp)
   4ea02:	00 
   4ea03:	c6 84 24 7c 01 00 00 	movb   $0x0,0x17c(%rsp)
   4ea0a:	00 
   4ea0b:	c6 84 24 84 01 00 00 	movb   $0x0,0x184(%rsp)
   4ea12:	00 
   4ea13:	c6 84 24 90 01 00 00 	movb   $0x0,0x190(%rsp)
   4ea1a:	00 
   4ea1b:	c6 84 24 a8 01 00 00 	movb   $0x0,0x1a8(%rsp)
   4ea22:	00 
   4ea23:	c6 84 24 c8 01 00 00 	movb   $0x0,0x1c8(%rsp)
   4ea2a:	00 
   4ea2b:	c6 84 24 d0 01 00 00 	movb   $0x1,0x1d0(%rsp)
   4ea32:	01 
   4ea33:	48 c7 84 24 98 00 00 	movq   $0x0,0x98(%rsp)
   4ea3a:	00 00 00 00 00 
			4ea3b: R_X86_64_32S	.rodata+0x48650
   4ea3f:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4ea44:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   4ea4b:	00 
   4ea4c:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ea4f:	0f 29 84 24 30 01 00 	movaps %xmm0,0x130(%rsp)
   4ea56:	00 
   4ea57:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ea5c:	e8 00 00 00 00       	call   4ea61 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23f1>
			4ea5d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4ea61:	31 c9                	xor    %ecx,%ecx
   4ea63:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4ea68:	0f 94 c1             	sete   %cl
   4ea6b:	0f 28 84 24 30 01 00 	movaps 0x130(%rsp),%xmm0
   4ea72:	00 
   4ea73:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ea77:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4ea7b:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4ea80:	48 8d bc 24 f8 03 00 	lea    0x3f8(%rsp),%rdi
   4ea87:	00 
   4ea88:	48 8d 94 24 a8 00 00 	lea    0xa8(%rsp),%rdx
   4ea8f:	00 
   4ea90:	48 8d 8c 24 98 00 00 	lea    0x98(%rsp),%rcx
   4ea97:	00 
   4ea98:	4c 8d 8c 24 40 01 00 	lea    0x140(%rsp),%r9
   4ea9f:	00 
   4eaa0:	48 89 de             	mov    %rbx,%rsi
   4eaa3:	4d 89 f0             	mov    %r14,%r8
   4eaa6:	e8 00 00 00 00       	call   4eaab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x243b>
			4eaa7: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4eaab:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4eab0:	e8 00 00 00 00       	call   4eab5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2445>
			4eab1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4eab5:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4eabc:	00 
   4eabd:	e8 00 00 00 00       	call   4eac2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2452>
			4eabe: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4eac2:	48 8b 9c 24 00 04 00 	mov    0x400(%rsp),%rbx
   4eac9:	00 
   4eaca:	48 85 db             	test   %rbx,%rbx
   4eacd:	74 5b                	je     4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4eacf:	48 8b 43 08          	mov    0x8(%rbx),%rax
   4ead3:	48 b9 01 00 00 00 01 	movabs $0x100000001,%rcx
   4eada:	00 00 00 
   4eadd:	48 39 c8             	cmp    %rcx,%rax
   4eae0:	75 1c                	jne    4eafe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x248e>
   4eae2:	48 c7 43 08 00 00 00 	movq   $0x0,0x8(%rbx)
   4eae9:	00 
   4eaea:	48 8b 03             	mov    (%rbx),%rax
   4eaed:	48 89 df             	mov    %rbx,%rdi
   4eaf0:	ff 50 10             	call   *0x10(%rax)
   4eaf3:	48 8b 03             	mov    (%rbx),%rax
   4eaf6:	48 89 df             	mov    %rbx,%rdi
   4eaf9:	ff 50 18             	call   *0x18(%rax)
   4eafc:	eb 2c                	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4eafe:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4eb05 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2495>
			4eb00: R_X86_64_PC32	__libc_single_threaded-0x5
   4eb05:	74 10                	je     4eb17 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a7>
   4eb07:	8d 48 ff             	lea    -0x1(%rax),%ecx
   4eb0a:	89 4b 08             	mov    %ecx,0x8(%rbx)
   4eb0d:	83 f8 01             	cmp    $0x1,%eax
   4eb10:	75 18                	jne    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4eb12:	e9 74 03 00 00       	jmp    4ee8b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x281b>
   4eb17:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
   4eb1c:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
   4eb21:	83 f8 01             	cmp    $0x1,%eax
   4eb24:	0f 84 61 03 00 00    	je     4ee8b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x281b>
   4eb2a:	48 81 c4 c8 04 00 00 	add    $0x4c8,%rsp
   4eb31:	5b                   	pop    %rbx
   4eb32:	41 5c                	pop    %r12
   4eb34:	41 5d                	pop    %r13
   4eb36:	41 5e                	pop    %r14
   4eb38:	41 5f                	pop    %r15
   4eb3a:	5d                   	pop    %rbp
   4eb3b:	c3                   	ret
   4eb3c:	81 f9 00 18 00 00    	cmp    $0x1800,%ecx
   4eb42:	75 09                	jne    4eb4d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24dd>
   4eb44:	41 81 f8 00 0a 00 00 	cmp    $0xa00,%r8d
   4eb4b:	74 19                	je     4eb66 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24f6>
   4eb4d:	81 f9 00 0a 00 00    	cmp    $0xa00,%ecx
   4eb53:	0f 85 4c de ff ff    	jne    4c9a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x335>
   4eb59:	41 81 f8 00 28 00 00 	cmp    $0x2800,%r8d
   4eb60:	0f 85 3f de ff ff    	jne    4c9a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x335>
   4eb66:	44 89 c5             	mov    %r8d,%ebp
   4eb69:	41 89 cd             	mov    %ecx,%r13d
   4eb6c:	49 89 d6             	mov    %rdx,%r14
   4eb6f:	48 89 5c 24 60       	mov    %rbx,0x60(%rsp)
   4eb74:	48 89 f3             	mov    %rsi,%rbx
   4eb77:	49 89 fc             	mov    %rdi,%r12
   4eb7a:	e8 41 66 03 00       	call   851c0 <_ZN6strata7kernels12_GLOBAL__N_110q6k_centerEv>
   4eb7f:	4c 89 e7             	mov    %r12,%rdi
   4eb82:	48 89 de             	mov    %rbx,%rsi
   4eb85:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
   4eb8a:	4c 89 f2             	mov    %r14,%rdx
   4eb8d:	44 89 e9             	mov    %r13d,%ecx
   4eb90:	41 89 e8             	mov    %ebp,%r8d
   4eb93:	84 c0                	test   %al,%al
   4eb95:	0f 84 0a de ff ff    	je     4c9a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x335>
   4eb9b:	49 c1 e7 03          	shl    $0x3,%r15
   4eb9f:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4eba6:	00 00 00 
   4eba9:	4c 21 f8             	and    %r15,%rax
   4ebac:	48 c7 84 24 68 02 00 	movq   $0x1,0x268(%rsp)
   4ebb3:	00 01 00 00 00 
   4ebb8:	48 c7 84 24 70 02 00 	movq   $0x4,0x270(%rsp)
   4ebbf:	00 04 00 00 00 
   4ebc4:	48 89 84 24 78 02 00 	mov    %rax,0x278(%rsp)
   4ebcb:	00 
   4ebcc:	48 c7 84 24 80 02 00 	movq   $0x1,0x280(%rsp)
   4ebd3:	00 01 00 00 00 
   4ebd8:	48 c7 84 24 88 02 00 	movq   $0x4,0x288(%rsp)
   4ebdf:	00 04 00 00 00 
   4ebe4:	48 c7 84 24 90 02 00 	movq   $0x20,0x290(%rsp)
   4ebeb:	00 20 00 00 00 
   4ebf0:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ebf3:	0f 11 84 24 98 02 00 	movups %xmm0,0x298(%rsp)
   4ebfa:	00 
   4ebfb:	48 c7 84 24 a8 02 00 	movq   $0x0,0x2a8(%rsp)
   4ec02:	00 00 00 00 00 
   4ec07:	48 89 bc 24 40 01 00 	mov    %rdi,0x140(%rsp)
   4ec0e:	00 
   4ec0f:	48 89 b4 24 48 01 00 	mov    %rsi,0x148(%rsp)
   4ec16:	00 
   4ec17:	48 89 94 24 50 01 00 	mov    %rdx,0x150(%rsp)
   4ec1e:	00 
   4ec1f:	89 8c 24 58 01 00 00 	mov    %ecx,0x158(%rsp)
   4ec26:	44 89 84 24 5c 01 00 	mov    %r8d,0x15c(%rsp)
   4ec2d:	00 
   4ec2e:	48 8b 84 24 a8 02 00 	mov    0x2a8(%rsp),%rax
   4ec35:	00 
   4ec36:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   4ec3b:	0f 10 84 24 68 02 00 	movups 0x268(%rsp),%xmm0
   4ec42:	00 
   4ec43:	0f 10 8c 24 78 02 00 	movups 0x278(%rsp),%xmm1
   4ec4a:	00 
   4ec4b:	0f 10 94 24 88 02 00 	movups 0x288(%rsp),%xmm2
   4ec52:	00 
   4ec53:	0f 10 9c 24 98 02 00 	movups 0x298(%rsp),%xmm3
   4ec5a:	00 
   4ec5b:	0f 11 5c 24 30       	movups %xmm3,0x30(%rsp)
   4ec60:	0f 11 54 24 20       	movups %xmm2,0x20(%rsp)
   4ec65:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
   4ec6a:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ec6e:	4c 8d b4 24 b8 04 00 	lea    0x4b8(%rsp),%r14
   4ec75:	00 
   4ec76:	48 8d 94 24 40 01 00 	lea    0x140(%rsp),%rdx
   4ec7d:	00 
   4ec7e:	4c 89 f7             	mov    %r14,%rdi
   4ec81:	48 89 de             	mov    %rbx,%rsi
   4ec84:	e8 87 65 03 00       	call   85210 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_123native_mmvq_q6k_center6EEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSB_PfiiPS1_EUlNS0_7nd_itemILi3EEEE7_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSI_>
   4ec89:	e9 f0 01 00 00       	jmp    4ee7e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x280e>
   4ec8e:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4ec95 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2625>
			4ec91: R_X86_64_PC32	.bss+0x204
   4ec95:	84 c0                	test   %al,%al
   4ec97:	0f 84 6a 0a 00 00    	je     4f707 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3097>
   4ec9d:	4e 8d 0c fd 18 00 00 	lea    0x18(,%r15,8),%r9
   4eca4:	00 
   4eca5:	48 b8 e0 ff ff ff 1f 	movabs $0x1fffffffe0,%rax
   4ecac:	00 00 00 
   4ecaf:	4c 21 c8             	and    %r9,%rax
   4ecb2:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4ecb9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2649>
			4ecb4: R_X86_64_PC32	.bss+0x1fb
   4ecb9:	0f 84 e2 00 00 00    	je     4eda1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2731>
   4ecbf:	48 c7 84 24 20 02 00 	movq   $0x1,0x220(%rsp)
   4ecc6:	00 01 00 00 00 
   4eccb:	48 c7 84 24 28 02 00 	movq   $0x4,0x228(%rsp)
   4ecd2:	00 04 00 00 00 
   4ecd7:	48 89 84 24 30 02 00 	mov    %rax,0x230(%rsp)
   4ecde:	00 
   4ecdf:	48 c7 84 24 38 02 00 	movq   $0x1,0x238(%rsp)
   4ece6:	00 01 00 00 00 
   4eceb:	48 c7 84 24 40 02 00 	movq   $0x4,0x240(%rsp)
   4ecf2:	00 04 00 00 00 
   4ecf7:	48 c7 84 24 48 02 00 	movq   $0x20,0x248(%rsp)
   4ecfe:	00 20 00 00 00 
   4ed03:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ed06:	0f 11 84 24 50 02 00 	movups %xmm0,0x250(%rsp)
   4ed0d:	00 
   4ed0e:	48 c7 84 24 60 02 00 	movq   $0x0,0x260(%rsp)
   4ed15:	00 00 00 00 00 
   4ed1a:	48 89 bc 24 40 01 00 	mov    %rdi,0x140(%rsp)
   4ed21:	00 
   4ed22:	48 89 b4 24 48 01 00 	mov    %rsi,0x148(%rsp)
   4ed29:	00 
   4ed2a:	48 89 94 24 50 01 00 	mov    %rdx,0x150(%rsp)
   4ed31:	00 
   4ed32:	89 8c 24 58 01 00 00 	mov    %ecx,0x158(%rsp)
   4ed39:	44 89 84 24 5c 01 00 	mov    %r8d,0x15c(%rsp)
   4ed40:	00 
   4ed41:	48 8b 84 24 60 02 00 	mov    0x260(%rsp),%rax
   4ed48:	00 
   4ed49:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   4ed4e:	0f 10 84 24 20 02 00 	movups 0x220(%rsp),%xmm0
   4ed55:	00 
   4ed56:	0f 10 8c 24 30 02 00 	movups 0x230(%rsp),%xmm1
   4ed5d:	00 
   4ed5e:	0f 10 94 24 40 02 00 	movups 0x240(%rsp),%xmm2
   4ed65:	00 
   4ed66:	0f 10 9c 24 50 02 00 	movups 0x250(%rsp),%xmm3
   4ed6d:	00 
   4ed6e:	0f 11 5c 24 30       	movups %xmm3,0x30(%rsp)
   4ed73:	0f 11 54 24 20       	movups %xmm2,0x20(%rsp)
   4ed78:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
   4ed7d:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ed81:	4c 8d b4 24 a8 04 00 	lea    0x4a8(%rsp),%r14
   4ed88:	00 
   4ed89:	48 8d 94 24 40 01 00 	lea    0x140(%rsp),%rdx
   4ed90:	00 
   4ed91:	4c 89 f7             	mov    %r14,%rdi
   4ed94:	48 89 de             	mov    %rbx,%rsi
   4ed97:	e8 d4 66 03 00       	call   85470 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_124native_mmvq_q6k_wide_a2rE18dpct_kernel_scalarILi6EES8_ILi1EEEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSE_PfiiPS1_EUlNS0_7nd_itemILi3EEEE10_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSL_>
   4ed9c:	e9 dd 00 00 00       	jmp    4ee7e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x280e>
   4eda1:	48 c7 84 24 d8 01 00 	movq   $0x1,0x1d8(%rsp)
   4eda8:	00 01 00 00 00 
   4edad:	48 c7 84 24 e0 01 00 	movq   $0x4,0x1e0(%rsp)
   4edb4:	00 04 00 00 00 
   4edb9:	48 89 84 24 e8 01 00 	mov    %rax,0x1e8(%rsp)
   4edc0:	00 
   4edc1:	48 c7 84 24 f0 01 00 	movq   $0x1,0x1f0(%rsp)
   4edc8:	00 01 00 00 00 
   4edcd:	48 c7 84 24 f8 01 00 	movq   $0x4,0x1f8(%rsp)
   4edd4:	00 04 00 00 00 
   4edd9:	48 c7 84 24 00 02 00 	movq   $0x20,0x200(%rsp)
   4ede0:	00 20 00 00 00 
   4ede5:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ede8:	0f 11 84 24 08 02 00 	movups %xmm0,0x208(%rsp)
   4edef:	00 
   4edf0:	48 c7 84 24 18 02 00 	movq   $0x0,0x218(%rsp)
   4edf7:	00 00 00 00 00 
   4edfc:	48 89 bc 24 40 01 00 	mov    %rdi,0x140(%rsp)
   4ee03:	00 
   4ee04:	48 89 b4 24 48 01 00 	mov    %rsi,0x148(%rsp)
   4ee0b:	00 
   4ee0c:	48 89 94 24 50 01 00 	mov    %rdx,0x150(%rsp)
   4ee13:	00 
   4ee14:	89 8c 24 58 01 00 00 	mov    %ecx,0x158(%rsp)
   4ee1b:	44 89 84 24 5c 01 00 	mov    %r8d,0x15c(%rsp)
   4ee22:	00 
   4ee23:	48 8b 84 24 18 02 00 	mov    0x218(%rsp),%rax
   4ee2a:	00 
   4ee2b:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   4ee30:	0f 10 84 24 d8 01 00 	movups 0x1d8(%rsp),%xmm0
   4ee37:	00 
   4ee38:	0f 10 8c 24 e8 01 00 	movups 0x1e8(%rsp),%xmm1
   4ee3f:	00 
   4ee40:	0f 10 94 24 f8 01 00 	movups 0x1f8(%rsp),%xmm2
   4ee47:	00 
   4ee48:	0f 10 9c 24 08 02 00 	movups 0x208(%rsp),%xmm3
   4ee4f:	00 
   4ee50:	0f 11 5c 24 30       	movups %xmm3,0x30(%rsp)
   4ee55:	0f 11 54 24 20       	movups %xmm2,0x20(%rsp)
   4ee5a:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
   4ee5f:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ee63:	4c 8d b4 24 98 04 00 	lea    0x498(%rsp),%r14
   4ee6a:	00 
   4ee6b:	48 8d 94 24 40 01 00 	lea    0x140(%rsp),%rdx
   4ee72:	00 
   4ee73:	4c 89 f7             	mov    %r14,%rdi
   4ee76:	48 89 de             	mov    %rbx,%rsi
   4ee79:	e8 52 68 03 00       	call   856d0 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_120native_mmvq_q6k_wideE18dpct_kernel_scalarILi6EES8_ILi1EEEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSE_PfiiPS1_EUlNS0_7nd_itemILi3EEEE11_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSL_>
   4ee7e:	4c 89 f7             	mov    %r14,%rdi
   4ee81:	e8 00 00 00 00       	call   4ee86 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2816>
			4ee82: R_X86_64_PLT32	_ZN4sycl3_V15eventD2Ev-0x4
   4ee86:	e9 9f fc ff ff       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4ee8b:	48 89 df             	mov    %rbx,%rdi
   4ee8e:	e8 00 00 00 00       	call   4ee93 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2823>
			4ee8f: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
   4ee93:	e9 92 fc ff ff       	jmp    4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>
   4ee98:	49 89 fe             	mov    %rdi,%r14
   4ee9b:	bf 00 00 00 00       	mov    $0x0,%edi
			4ee9c: R_X86_64_32	.bss+0x1948
   4eea0:	44 89 c5             	mov    %r8d,%ebp
   4eea3:	41 89 cf             	mov    %ecx,%r15d
   4eea6:	49 89 d4             	mov    %rdx,%r12
   4eea9:	49 89 f5             	mov    %rsi,%r13
   4eeac:	e8 00 00 00 00       	call   4eeb1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2841>
			4eead: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4eeb1:	4c 89 f7             	mov    %r14,%rdi
   4eeb4:	4c 89 ee             	mov    %r13,%rsi
   4eeb7:	4c 89 e2             	mov    %r12,%rdx
   4eeba:	44 89 f9             	mov    %r15d,%ecx
   4eebd:	41 89 e8             	mov    %ebp,%r8d
   4eec0:	85 c0                	test   %eax,%eax
   4eec2:	0f 84 cb d7 ff ff    	je     4c693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23>
   4eec8:	bf 00 00 00 00       	mov    $0x0,%edi
			4eec9: R_X86_64_32	.rodata.str1.1+0x1244d
   4eecd:	e8 00 00 00 00       	call   4eed2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2862>
			4eece: R_X86_64_PLT32	getenv-0x4
   4eed2:	48 85 c0             	test   %rax,%rax
   4eed5:	74 65                	je     4ef3c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28cc>
   4eed7:	48 89 c7             	mov    %rax,%rdi
   4eeda:	31 f6                	xor    %esi,%esi
   4eedc:	ba 0a 00 00 00       	mov    $0xa,%edx
   4eee1:	e8 00 00 00 00       	call   4eee6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2876>
			4eee2: R_X86_64_PLT32	__isoc23_strtol-0x4
   4eee6:	eb 59                	jmp    4ef41 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28d1>
   4eee8:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4eeed:	bf 00 00 00 00       	mov    $0x0,%edi
			4eeee: R_X86_64_32	.bss+0x1958
   4eef2:	45 89 c7             	mov    %r8d,%r15d
   4eef5:	41 89 cc             	mov    %ecx,%r12d
   4eef8:	49 89 d5             	mov    %rdx,%r13
   4eefb:	49 89 f6             	mov    %rsi,%r14
   4eefe:	e8 00 00 00 00       	call   4ef03 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2893>
			4eeff: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ef03:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ef08:	4c 89 f6             	mov    %r14,%rsi
   4ef0b:	4c 89 ea             	mov    %r13,%rdx
   4ef0e:	44 89 e1             	mov    %r12d,%ecx
   4ef11:	45 89 f8             	mov    %r15d,%r8d
   4ef14:	85 c0                	test   %eax,%eax
   4ef16:	0f 84 8c d7 ff ff    	je     4c6a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x38>
   4ef1c:	bf 00 00 00 00       	mov    $0x0,%edi
			4ef1d: R_X86_64_32	.rodata.str1.1+0x12462
   4ef21:	e8 00 00 00 00       	call   4ef26 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28b6>
			4ef22: R_X86_64_PLT32	getenv-0x4
   4ef26:	48 85 c0             	test   %rax,%rax
   4ef29:	74 3a                	je     4ef65 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28f5>
   4ef2b:	48 89 c7             	mov    %rax,%rdi
   4ef2e:	31 f6                	xor    %esi,%esi
   4ef30:	ba 0a 00 00 00       	mov    $0xa,%edx
   4ef35:	e8 00 00 00 00       	call   4ef3a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28ca>
			4ef36: R_X86_64_PLT32	__isoc23_strtol-0x4
   4ef3a:	eb 2b                	jmp    4ef67 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28f7>
   4ef3c:	b8 01 00 00 00       	mov    $0x1,%eax
   4ef41:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4ef47 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28d7>
			4ef43: R_X86_64_PC32	.bss+0x193c
   4ef47:	bf 00 00 00 00       	mov    $0x0,%edi
			4ef48: R_X86_64_32	.bss+0x1948
   4ef4c:	e8 00 00 00 00       	call   4ef51 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28e1>
			4ef4d: R_X86_64_PLT32	__cxa_guard_release-0x4
   4ef51:	41 89 e8             	mov    %ebp,%r8d
   4ef54:	44 89 f9             	mov    %r15d,%ecx
   4ef57:	4c 89 e2             	mov    %r12,%rdx
   4ef5a:	4c 89 ee             	mov    %r13,%rsi
   4ef5d:	4c 89 f7             	mov    %r14,%rdi
   4ef60:	e9 2e d7 ff ff       	jmp    4c693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23>
   4ef65:	31 c0                	xor    %eax,%eax
   4ef67:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4ef6d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28fd>
			4ef69: R_X86_64_PC32	.bss+0x194c
   4ef6d:	bf 00 00 00 00       	mov    $0x0,%edi
			4ef6e: R_X86_64_32	.bss+0x1958
   4ef72:	e8 00 00 00 00       	call   4ef77 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2907>
			4ef73: R_X86_64_PLT32	__cxa_guard_release-0x4
   4ef77:	45 89 f8             	mov    %r15d,%r8d
   4ef7a:	44 89 e1             	mov    %r12d,%ecx
   4ef7d:	4c 89 ea             	mov    %r13,%rdx
   4ef80:	4c 89 f6             	mov    %r14,%rsi
   4ef83:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ef88:	e9 1b d7 ff ff       	jmp    4c6a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x38>
   4ef8d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4ef92:	bf 00 00 00 00       	mov    $0x0,%edi
			4ef93: R_X86_64_32	.bss+0x208
   4ef97:	44 89 c5             	mov    %r8d,%ebp
   4ef9a:	41 89 cc             	mov    %ecx,%r12d
   4ef9d:	49 89 d5             	mov    %rdx,%r13
   4efa0:	49 89 f6             	mov    %rsi,%r14
   4efa3:	e8 00 00 00 00       	call   4efa8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2938>
			4efa4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4efa8:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4efad:	4c 89 f6             	mov    %r14,%rsi
   4efb0:	4c 89 ea             	mov    %r13,%rdx
   4efb3:	44 89 e1             	mov    %r12d,%ecx
   4efb6:	41 89 e8             	mov    %ebp,%r8d
   4efb9:	85 c0                	test   %eax,%eax
   4efbb:	0f 84 32 dc ff ff    	je     4cbf3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x583>
   4efc1:	bf 00 00 00 00       	mov    $0x0,%edi
			4efc2: R_X86_64_32	.rodata.str1.1+0x1cb3
   4efc6:	e8 00 00 00 00       	call   4efcb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x295b>
			4efc7: R_X86_64_PLT32	getenv-0x4
   4efcb:	48 85 c0             	test   %rax,%rax
   4efce:	0f 84 f5 05 00 00    	je     4f5c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f59>
   4efd4:	48 89 c7             	mov    %rax,%rdi
   4efd7:	31 f6                	xor    %esi,%esi
   4efd9:	ba 0a 00 00 00       	mov    $0xa,%edx
   4efde:	e8 00 00 00 00       	call   4efe3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2973>
			4efdf: R_X86_64_PLT32	__isoc23_strtol-0x4
   4efe3:	85 c0                	test   %eax,%eax
   4efe5:	0f 95 c0             	setne  %al
   4efe8:	e9 de 05 00 00       	jmp    4f5cb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f5b>
   4efed:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4eff2:	bf 00 00 00 00       	mov    $0x0,%edi
			4eff3: R_X86_64_32	.bss+0x208
   4eff7:	44 89 c5             	mov    %r8d,%ebp
   4effa:	41 89 cc             	mov    %ecx,%r12d
   4effd:	49 89 d5             	mov    %rdx,%r13
   4f000:	49 89 f6             	mov    %rsi,%r14
   4f003:	e8 00 00 00 00       	call   4f008 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2998>
			4f004: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f008:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f00d:	4c 89 f6             	mov    %r14,%rsi
   4f010:	4c 89 ea             	mov    %r13,%rdx
   4f013:	44 89 e1             	mov    %r12d,%ecx
   4f016:	41 89 e8             	mov    %ebp,%r8d
   4f019:	85 c0                	test   %eax,%eax
   4f01b:	0f 84 be d6 ff ff    	je     4c6df <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6f>
   4f021:	bf 00 00 00 00       	mov    $0x0,%edi
			4f022: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f026:	e8 00 00 00 00       	call   4f02b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29bb>
			4f027: R_X86_64_PLT32	getenv-0x4
   4f02b:	48 85 c0             	test   %rax,%rax
   4f02e:	0f 84 bd 05 00 00    	je     4f5f1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f81>
   4f034:	48 89 c7             	mov    %rax,%rdi
   4f037:	31 f6                	xor    %esi,%esi
   4f039:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f03e:	e8 00 00 00 00       	call   4f043 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29d3>
			4f03f: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f043:	85 c0                	test   %eax,%eax
   4f045:	0f 95 c0             	setne  %al
   4f048:	e9 a6 05 00 00       	jmp    4f5f3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f83>
   4f04d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f052:	bf 00 00 00 00       	mov    $0x0,%edi
			4f053: R_X86_64_32	.bss+0x208
   4f057:	44 89 c5             	mov    %r8d,%ebp
   4f05a:	41 89 cc             	mov    %ecx,%r12d
   4f05d:	49 89 d5             	mov    %rdx,%r13
   4f060:	49 89 f6             	mov    %rsi,%r14
   4f063:	e8 00 00 00 00       	call   4f068 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29f8>
			4f064: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f068:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f06d:	4c 89 f6             	mov    %r14,%rsi
   4f070:	4c 89 ea             	mov    %r13,%rdx
   4f073:	44 89 e1             	mov    %r12d,%ecx
   4f076:	41 89 e8             	mov    %ebp,%r8d
   4f079:	85 c0                	test   %eax,%eax
   4f07b:	0f 84 ed dd ff ff    	je     4ce6e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7fe>
   4f081:	bf 00 00 00 00       	mov    $0x0,%edi
			4f082: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f086:	e8 00 00 00 00       	call   4f08b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a1b>
			4f087: R_X86_64_PLT32	getenv-0x4
   4f08b:	48 85 c0             	test   %rax,%rax
   4f08e:	0f 84 85 05 00 00    	je     4f619 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fa9>
   4f094:	48 89 c7             	mov    %rax,%rdi
   4f097:	31 f6                	xor    %esi,%esi
   4f099:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f09e:	e8 00 00 00 00       	call   4f0a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a33>
			4f09f: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f0a3:	85 c0                	test   %eax,%eax
   4f0a5:	0f 95 c0             	setne  %al
   4f0a8:	e9 6e 05 00 00       	jmp    4f61b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fab>
   4f0ad:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f0b2:	bf 00 00 00 00       	mov    $0x0,%edi
			4f0b3: R_X86_64_32	.bss+0x208
   4f0b7:	44 89 c5             	mov    %r8d,%ebp
   4f0ba:	41 89 cc             	mov    %ecx,%r12d
   4f0bd:	49 89 d5             	mov    %rdx,%r13
   4f0c0:	49 89 f6             	mov    %rsi,%r14
   4f0c3:	e8 00 00 00 00       	call   4f0c8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a58>
			4f0c4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f0c8:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f0cd:	4c 89 f6             	mov    %r14,%rsi
   4f0d0:	4c 89 ea             	mov    %r13,%rdx
   4f0d3:	44 89 e1             	mov    %r12d,%ecx
   4f0d6:	41 89 e8             	mov    %ebp,%r8d
   4f0d9:	85 c0                	test   %eax,%eax
   4f0db:	0f 84 08 e0 ff ff    	je     4d0e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa79>
   4f0e1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f0e2: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f0e6:	e8 00 00 00 00       	call   4f0eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a7b>
			4f0e7: R_X86_64_PLT32	getenv-0x4
   4f0eb:	48 85 c0             	test   %rax,%rax
   4f0ee:	0f 84 4d 05 00 00    	je     4f641 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fd1>
   4f0f4:	48 89 c7             	mov    %rax,%rdi
   4f0f7:	31 f6                	xor    %esi,%esi
   4f0f9:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f0fe:	e8 00 00 00 00       	call   4f103 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a93>
			4f0ff: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f103:	85 c0                	test   %eax,%eax
   4f105:	0f 95 c0             	setne  %al
   4f108:	e9 36 05 00 00       	jmp    4f643 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fd3>
   4f10d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f112:	bf 00 00 00 00       	mov    $0x0,%edi
			4f113: R_X86_64_32	.bss+0x208
   4f117:	44 89 c5             	mov    %r8d,%ebp
   4f11a:	41 89 cc             	mov    %ecx,%r12d
   4f11d:	49 89 d5             	mov    %rdx,%r13
   4f120:	49 89 f6             	mov    %rsi,%r14
   4f123:	e8 00 00 00 00       	call   4f128 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ab8>
			4f124: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f128:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f12d:	4c 89 f6             	mov    %r14,%rsi
   4f130:	4c 89 ea             	mov    %r13,%rdx
   4f133:	44 89 e1             	mov    %r12d,%ecx
   4f136:	41 89 e8             	mov    %ebp,%r8d
   4f139:	85 c0                	test   %eax,%eax
   4f13b:	0f 84 26 e2 ff ff    	je     4d367 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcf7>
   4f141:	bf 00 00 00 00       	mov    $0x0,%edi
			4f142: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f146:	e8 00 00 00 00       	call   4f14b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2adb>
			4f147: R_X86_64_PLT32	getenv-0x4
   4f14b:	48 85 c0             	test   %rax,%rax
   4f14e:	0f 84 15 05 00 00    	je     4f669 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ff9>
   4f154:	48 89 c7             	mov    %rax,%rdi
   4f157:	31 f6                	xor    %esi,%esi
   4f159:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f15e:	e8 00 00 00 00       	call   4f163 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2af3>
			4f15f: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f163:	85 c0                	test   %eax,%eax
   4f165:	0f 95 c0             	setne  %al
   4f168:	e9 fe 04 00 00       	jmp    4f66b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ffb>
   4f16d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f172:	bf 00 00 00 00       	mov    $0x0,%edi
			4f173: R_X86_64_32	.bss+0x208
   4f177:	44 89 c5             	mov    %r8d,%ebp
   4f17a:	41 89 cc             	mov    %ecx,%r12d
   4f17d:	49 89 d5             	mov    %rdx,%r13
   4f180:	49 89 f6             	mov    %rsi,%r14
   4f183:	e8 00 00 00 00       	call   4f188 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b18>
			4f184: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f188:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f18d:	4c 89 f6             	mov    %r14,%rsi
   4f190:	4c 89 ea             	mov    %r13,%rdx
   4f193:	44 89 e1             	mov    %r12d,%ecx
   4f196:	41 89 e8             	mov    %ebp,%r8d
   4f199:	85 c0                	test   %eax,%eax
   4f19b:	0f 84 28 e4 ff ff    	je     4d5c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf59>
   4f1a1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f1a2: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f1a6:	e8 00 00 00 00       	call   4f1ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b3b>
			4f1a7: R_X86_64_PLT32	getenv-0x4
   4f1ab:	48 85 c0             	test   %rax,%rax
   4f1ae:	0f 84 dd 04 00 00    	je     4f691 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3021>
   4f1b4:	48 89 c7             	mov    %rax,%rdi
   4f1b7:	31 f6                	xor    %esi,%esi
   4f1b9:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f1be:	e8 00 00 00 00       	call   4f1c3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b53>
			4f1bf: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f1c3:	85 c0                	test   %eax,%eax
   4f1c5:	0f 95 c0             	setne  %al
   4f1c8:	e9 c6 04 00 00       	jmp    4f693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3023>
   4f1cd:	49 89 fe             	mov    %rdi,%r14
   4f1d0:	bf 00 00 00 00       	mov    $0x0,%edi
			4f1d1: R_X86_64_32	.bss+0x208
   4f1d5:	44 89 c5             	mov    %r8d,%ebp
   4f1d8:	41 89 cf             	mov    %ecx,%r15d
   4f1db:	49 89 d4             	mov    %rdx,%r12
   4f1de:	49 89 f5             	mov    %rsi,%r13
   4f1e1:	e8 00 00 00 00       	call   4f1e6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b76>
			4f1e2: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f1e6:	4c 89 f7             	mov    %r14,%rdi
   4f1e9:	4c 89 ee             	mov    %r13,%rsi
   4f1ec:	4c 89 e2             	mov    %r12,%rdx
   4f1ef:	44 89 f9             	mov    %r15d,%ecx
   4f1f2:	41 89 e8             	mov    %ebp,%r8d
   4f1f5:	85 c0                	test   %eax,%eax
   4f1f7:	0f 84 78 d7 ff ff    	je     4c975 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x305>
   4f1fd:	bf 00 00 00 00       	mov    $0x0,%edi
			4f1fe: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f202:	e8 00 00 00 00       	call   4f207 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b97>
			4f203: R_X86_64_PLT32	getenv-0x4
   4f207:	48 85 c0             	test   %rax,%rax
   4f20a:	0f 84 a9 04 00 00    	je     4f6b9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3049>
   4f210:	48 89 c7             	mov    %rax,%rdi
   4f213:	31 f6                	xor    %esi,%esi
   4f215:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f21a:	e8 00 00 00 00       	call   4f21f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2baf>
			4f21b: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f21f:	85 c0                	test   %eax,%eax
   4f221:	0f 95 c0             	setne  %al
   4f224:	e9 92 04 00 00       	jmp    4f6bb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x304b>
   4f229:	bf 00 00 00 00       	mov    $0x0,%edi
			4f22a: R_X86_64_32	.bss+0x1fd8
   4f22e:	e8 00 00 00 00       	call   4f233 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bc3>
			4f22f: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f233:	85 c0                	test   %eax,%eax
   4f235:	0f 84 12 e7 ff ff    	je     4d94d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12dd>
   4f23b:	bf 00 00 00 00       	mov    $0x0,%edi
			4f23c: R_X86_64_32	.rodata+0x49420
   4f240:	e8 00 00 00 00       	call   4f245 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bd5>
			4f241: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f245:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f24c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bdc>
			4f248: R_X86_64_PC32	.bss+0x1fcc
   4f24c:	bf 00 00 00 00       	mov    $0x0,%edi
			4f24d: R_X86_64_32	.bss+0x1fd8
   4f251:	e8 00 00 00 00       	call   4f256 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2be6>
			4f252: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f256:	e9 f2 e6 ff ff       	jmp    4d94d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12dd>
   4f25b:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f260:	bf 00 00 00 00       	mov    $0x0,%edi
			4f261: R_X86_64_32	.bss+0x1958
   4f265:	44 89 c5             	mov    %r8d,%ebp
   4f268:	41 89 cc             	mov    %ecx,%r12d
   4f26b:	49 89 d5             	mov    %rdx,%r13
   4f26e:	49 89 f6             	mov    %rsi,%r14
   4f271:	e8 00 00 00 00       	call   4f276 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c06>
			4f272: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f276:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f27b:	4c 89 f6             	mov    %r14,%rsi
   4f27e:	4c 89 ea             	mov    %r13,%rdx
   4f281:	44 89 e1             	mov    %r12d,%ecx
   4f284:	41 89 e8             	mov    %ebp,%r8d
   4f287:	85 c0                	test   %eax,%eax
   4f289:	0f 84 05 d7 ff ff    	je     4c994 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x324>
   4f28f:	bf 00 00 00 00       	mov    $0x0,%edi
			4f290: R_X86_64_32	.rodata.str1.1+0x12462
   4f294:	e8 00 00 00 00       	call   4f299 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c29>
			4f295: R_X86_64_PLT32	getenv-0x4
   4f299:	48 85 c0             	test   %rax,%rax
   4f29c:	0f 84 3d 04 00 00    	je     4f6df <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x306f>
   4f2a2:	48 89 c7             	mov    %rax,%rdi
   4f2a5:	31 f6                	xor    %esi,%esi
   4f2a7:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f2ac:	e8 00 00 00 00       	call   4f2b1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c41>
			4f2ad: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f2b1:	e9 2b 04 00 00       	jmp    4f6e1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3071>
   4f2b6:	bf 00 00 00 00       	mov    $0x0,%edi
			4f2b7: R_X86_64_32	.bss+0x1ef8
   4f2bb:	e8 00 00 00 00       	call   4f2c0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c50>
			4f2bc: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f2c0:	85 c0                	test   %eax,%eax
   4f2c2:	0f 84 83 da ff ff    	je     4cd4b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6db>
   4f2c8:	bf 00 00 00 00       	mov    $0x0,%edi
			4f2c9: R_X86_64_32	.rodata+0x479b8
   4f2cd:	e8 00 00 00 00       	call   4f2d2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c62>
			4f2ce: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f2d2:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f2d9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c69>
			4f2d5: R_X86_64_PC32	.bss+0x1eec
   4f2d9:	bf 00 00 00 00       	mov    $0x0,%edi
			4f2da: R_X86_64_32	.bss+0x1ef8
   4f2de:	e8 00 00 00 00       	call   4f2e3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c73>
			4f2df: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f2e3:	e9 63 da ff ff       	jmp    4cd4b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6db>
   4f2e8:	bf 00 00 00 00       	mov    $0x0,%edi
			4f2e9: R_X86_64_32	.bss+0x1eb8
   4f2ed:	e8 00 00 00 00       	call   4f2f2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c82>
			4f2ee: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f2f2:	85 c0                	test   %eax,%eax
   4f2f4:	0f 84 3d d5 ff ff    	je     4c837 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c7>
   4f2fa:	bf 00 00 00 00       	mov    $0x0,%edi
			4f2fb: R_X86_64_32	.rodata+0x47198
   4f2ff:	e8 00 00 00 00       	call   4f304 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c94>
			4f300: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f304:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f30b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c9b>
			4f307: R_X86_64_PC32	.bss+0x1eac
   4f30b:	bf 00 00 00 00       	mov    $0x0,%edi
			4f30c: R_X86_64_32	.bss+0x1eb8
   4f310:	e8 00 00 00 00       	call   4f315 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ca5>
			4f311: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f315:	e9 1d d5 ff ff       	jmp    4c837 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c7>
   4f31a:	bf 00 00 00 00       	mov    $0x0,%edi
			4f31b: R_X86_64_32	.bss+0x1ed8
   4f31f:	e8 00 00 00 00       	call   4f324 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cb4>
			4f320: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f324:	85 c0                	test   %eax,%eax
   4f326:	0f 84 9a dc ff ff    	je     4cfc6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x956>
   4f32c:	bf 00 00 00 00       	mov    $0x0,%edi
			4f32d: R_X86_64_32	.rodata+0x475a8
   4f331:	e8 00 00 00 00       	call   4f336 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cc6>
			4f332: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f336:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f33d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ccd>
			4f339: R_X86_64_PC32	.bss+0x1ecc
   4f33d:	bf 00 00 00 00       	mov    $0x0,%edi
			4f33e: R_X86_64_32	.bss+0x1ed8
   4f342:	e8 00 00 00 00       	call   4f347 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cd7>
			4f343: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f347:	e9 7a dc ff ff       	jmp    4cfc6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x956>
   4f34c:	bf 00 00 00 00       	mov    $0x0,%edi
			4f34d: R_X86_64_32	.bss+0x1f18
   4f351:	e8 00 00 00 00       	call   4f356 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ce6>
			4f352: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f356:	85 c0                	test   %eax,%eax
   4f358:	0f 84 e3 de ff ff    	je     4d241 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbd1>
   4f35e:	bf 00 00 00 00       	mov    $0x0,%edi
			4f35f: R_X86_64_32	.rodata+0x47dc8
   4f363:	e8 00 00 00 00       	call   4f368 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cf8>
			4f364: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f368:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f36f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cff>
			4f36b: R_X86_64_PC32	.bss+0x1f0c
   4f36f:	bf 00 00 00 00       	mov    $0x0,%edi
			4f370: R_X86_64_32	.bss+0x1f18
   4f374:	e8 00 00 00 00       	call   4f379 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d09>
			4f375: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f379:	e9 c3 de ff ff       	jmp    4d241 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbd1>
   4f37e:	bf 00 00 00 00       	mov    $0x0,%edi
			4f37f: R_X86_64_32	.bss+0x1f98
   4f383:	e8 00 00 00 00       	call   4f388 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d18>
			4f384: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f388:	85 c0                	test   %eax,%eax
   4f38a:	0f 84 13 e1 ff ff    	je     4d4a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe33>
   4f390:	bf 00 00 00 00       	mov    $0x0,%edi
			4f391: R_X86_64_32	.rodata+0x48c00
   4f395:	e8 00 00 00 00       	call   4f39a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d2a>
			4f396: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f39a:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f3a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d31>
			4f39d: R_X86_64_PC32	.bss+0x1f8c
   4f3a1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3a2: R_X86_64_32	.bss+0x1f98
   4f3a6:	e8 00 00 00 00       	call   4f3ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d3b>
			4f3a7: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f3ab:	e9 f3 e0 ff ff       	jmp    4d4a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe33>
   4f3b0:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3b1: R_X86_64_32	.bss+0x1fb8
   4f3b5:	e8 00 00 00 00       	call   4f3ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d4a>
			4f3b6: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f3ba:	85 c0                	test   %eax,%eax
   4f3bc:	0f 84 42 e3 ff ff    	je     4d704 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1094>
   4f3c2:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3c3: R_X86_64_32	.rodata+0x49010
   4f3c7:	e8 00 00 00 00       	call   4f3cc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d5c>
			4f3c8: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f3cc:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f3d3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d63>
			4f3cf: R_X86_64_PC32	.bss+0x1fac
   4f3d3:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3d4: R_X86_64_32	.bss+0x1fb8
   4f3d8:	e8 00 00 00 00       	call   4f3dd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d6d>
			4f3d9: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f3dd:	e9 22 e3 ff ff       	jmp    4d704 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1094>
   4f3e2:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3e3: R_X86_64_32	.bss+0x1f58
   4f3e7:	e8 00 00 00 00       	call   4f3ec <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d7c>
			4f3e8: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f3ec:	85 c0                	test   %eax,%eax
   4f3ee:	0f 84 dc d6 ff ff    	je     4cad0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x460>
   4f3f4:	bf 00 00 00 00       	mov    $0x0,%edi
			4f3f5: R_X86_64_32	.rodata+0x483e0
   4f3f9:	e8 00 00 00 00       	call   4f3fe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d8e>
			4f3fa: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f3fe:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f405 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d95>
			4f401: R_X86_64_PC32	.bss+0x1f4c
   4f405:	bf 00 00 00 00       	mov    $0x0,%edi
			4f406: R_X86_64_32	.bss+0x1f58
   4f40a:	e8 00 00 00 00       	call   4f40f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d9f>
			4f40b: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f40f:	e9 bc d6 ff ff       	jmp    4cad0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x460>
   4f414:	49 89 fe             	mov    %rdi,%r14
   4f417:	bf 00 00 00 00       	mov    $0x0,%edi
			4f418: R_X86_64_32	.bss+0x4d8
   4f41c:	44 89 c5             	mov    %r8d,%ebp
   4f41f:	41 89 cf             	mov    %ecx,%r15d
   4f422:	49 89 d4             	mov    %rdx,%r12
   4f425:	49 89 f5             	mov    %rsi,%r13
   4f428:	e8 00 00 00 00       	call   4f42d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dbd>
			4f429: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f42d:	4c 89 f7             	mov    %r14,%rdi
   4f430:	4c 89 ee             	mov    %r13,%rsi
   4f433:	4c 89 e2             	mov    %r12,%rdx
   4f436:	44 89 f9             	mov    %r15d,%ecx
   4f439:	41 89 e8             	mov    %ebp,%r8d
   4f43c:	85 c0                	test   %eax,%eax
   4f43e:	0f 84 25 f4 ff ff    	je     4e869 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21f9>
   4f444:	bf 00 00 00 00       	mov    $0x0,%edi
			4f445: R_X86_64_32	.rodata.str1.1+0x3bcf
   4f449:	e8 00 00 00 00       	call   4f44e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dde>
			4f44a: R_X86_64_PLT32	getenv-0x4
   4f44e:	48 85 c0             	test   %rax,%rax
   4f451:	0f 84 09 03 00 00    	je     4f760 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30f0>
   4f457:	48 89 c7             	mov    %rax,%rdi
   4f45a:	31 f6                	xor    %esi,%esi
   4f45c:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f461:	e8 00 00 00 00       	call   4f466 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2df6>
			4f462: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f466:	e9 fa 02 00 00       	jmp    4f765 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30f5>
   4f46b:	bf 00 00 00 00       	mov    $0x0,%edi
			4f46c: R_X86_64_32	.bss+0x1f08
   4f470:	e8 00 00 00 00       	call   4f475 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e05>
			4f471: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f475:	85 c0                	test   %eax,%eax
   4f477:	0f 84 2f e7 ff ff    	je     4dbac <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x153c>
   4f47d:	bf 00 00 00 00       	mov    $0x0,%edi
			4f47e: R_X86_64_32	.rodata+0x47bc0
   4f482:	e8 00 00 00 00       	call   4f487 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e17>
			4f483: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f487:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f48e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e1e>
			4f48a: R_X86_64_PC32	.bss+0x1efc
   4f48e:	bf 00 00 00 00       	mov    $0x0,%edi
			4f48f: R_X86_64_32	.bss+0x1f08
   4f493:	e8 00 00 00 00       	call   4f498 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e28>
			4f494: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f498:	e9 0f e7 ff ff       	jmp    4dbac <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x153c>
   4f49d:	bf 00 00 00 00       	mov    $0x0,%edi
			4f49e: R_X86_64_32	.bss+0x1ec8
   4f4a2:	e8 00 00 00 00       	call   4f4a7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e37>
			4f4a3: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f4a7:	85 c0                	test   %eax,%eax
   4f4a9:	0f 84 5c e9 ff ff    	je     4de0b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x179b>
   4f4af:	bf 00 00 00 00       	mov    $0x0,%edi
			4f4b0: R_X86_64_32	.rodata+0x473a0
   4f4b4:	e8 00 00 00 00       	call   4f4b9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e49>
			4f4b5: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f4b9:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f4c0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e50>
			4f4bc: R_X86_64_PC32	.bss+0x1ebc
   4f4c0:	bf 00 00 00 00       	mov    $0x0,%edi
			4f4c1: R_X86_64_32	.bss+0x1ec8
   4f4c5:	e8 00 00 00 00       	call   4f4ca <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e5a>
			4f4c6: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f4ca:	e9 3c e9 ff ff       	jmp    4de0b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x179b>
   4f4cf:	bf 00 00 00 00       	mov    $0x0,%edi
			4f4d0: R_X86_64_32	.bss+0x1ee8
   4f4d4:	e8 00 00 00 00       	call   4f4d9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e69>
			4f4d5: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f4d9:	85 c0                	test   %eax,%eax
   4f4db:	0f 84 89 eb ff ff    	je     4e06a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19fa>
   4f4e1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f4e2: R_X86_64_32	.rodata+0x477b0
   4f4e6:	e8 00 00 00 00       	call   4f4eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e7b>
			4f4e7: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f4eb:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f4f2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e82>
			4f4ee: R_X86_64_PC32	.bss+0x1edc
   4f4f2:	bf 00 00 00 00       	mov    $0x0,%edi
			4f4f3: R_X86_64_32	.bss+0x1ee8
   4f4f7:	e8 00 00 00 00       	call   4f4fc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e8c>
			4f4f8: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f4fc:	e9 69 eb ff ff       	jmp    4e06a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19fa>
   4f501:	bf 00 00 00 00       	mov    $0x0,%edi
			4f502: R_X86_64_32	.bss+0x1f28
   4f506:	e8 00 00 00 00       	call   4f50b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e9b>
			4f507: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f50b:	85 c0                	test   %eax,%eax
   4f50d:	0f 84 b6 ed ff ff    	je     4e2c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c59>
   4f513:	bf 00 00 00 00       	mov    $0x0,%edi
			4f514: R_X86_64_32	.rodata+0x47fd0
   4f518:	e8 00 00 00 00       	call   4f51d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ead>
			4f519: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f51d:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f524 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2eb4>
			4f520: R_X86_64_PC32	.bss+0x1f1c
   4f524:	bf 00 00 00 00       	mov    $0x0,%edi
			4f525: R_X86_64_32	.bss+0x1f28
   4f529:	e8 00 00 00 00       	call   4f52e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ebe>
			4f52a: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f52e:	e9 96 ed ff ff       	jmp    4e2c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c59>
   4f533:	bf 00 00 00 00       	mov    $0x0,%edi
			4f534: R_X86_64_32	.bss+0x1fa8
   4f538:	e8 00 00 00 00       	call   4f53d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ecd>
			4f539: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f53d:	85 c0                	test   %eax,%eax
   4f53f:	0f 84 c3 ef ff ff    	je     4e508 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e98>
   4f545:	bf 00 00 00 00       	mov    $0x0,%edi
			4f546: R_X86_64_32	.rodata+0x48e08
   4f54a:	e8 00 00 00 00       	call   4f54f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2edf>
			4f54b: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f54f:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f556 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ee6>
			4f552: R_X86_64_PC32	.bss+0x1f9c
   4f556:	bf 00 00 00 00       	mov    $0x0,%edi
			4f557: R_X86_64_32	.bss+0x1fa8
   4f55b:	e8 00 00 00 00       	call   4f560 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ef0>
			4f55c: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f560:	e9 a3 ef ff ff       	jmp    4e508 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e98>
   4f565:	bf 00 00 00 00       	mov    $0x0,%edi
			4f566: R_X86_64_32	.bss+0x1fc8
   4f56a:	e8 00 00 00 00       	call   4f56f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2eff>
			4f56b: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f56f:	85 c0                	test   %eax,%eax
   4f571:	0f 84 cf f1 ff ff    	je     4e746 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20d6>
   4f577:	bf 00 00 00 00       	mov    $0x0,%edi
			4f578: R_X86_64_32	.rodata+0x49218
   4f57c:	e8 00 00 00 00       	call   4f581 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f11>
			4f57d: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f581:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f588 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f18>
			4f584: R_X86_64_PC32	.bss+0x1fbc
   4f588:	bf 00 00 00 00       	mov    $0x0,%edi
			4f589: R_X86_64_32	.bss+0x1fc8
   4f58d:	e8 00 00 00 00       	call   4f592 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f22>
			4f58e: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f592:	e9 af f1 ff ff       	jmp    4e746 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20d6>
   4f597:	bf 00 00 00 00       	mov    $0x0,%edi
			4f598: R_X86_64_32	.bss+0x1f68
   4f59c:	e8 00 00 00 00       	call   4f5a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f31>
			4f59d: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f5a1:	85 c0                	test   %eax,%eax
   4f5a3:	0f 84 1b f4 ff ff    	je     4e9c4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2354>
   4f5a9:	bf 00 00 00 00       	mov    $0x0,%edi
			4f5aa: R_X86_64_32	.rodata+0x485e8
   4f5ae:	e8 00 00 00 00       	call   4f5b3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f43>
			4f5af: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f5b3:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f5ba <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f4a>
			4f5b6: R_X86_64_PC32	.bss+0x1f5c
   4f5ba:	bf 00 00 00 00       	mov    $0x0,%edi
			4f5bb: R_X86_64_32	.bss+0x1f68
   4f5bf:	e8 00 00 00 00       	call   4f5c4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f54>
			4f5c0: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f5c4:	e9 fb f3 ff ff       	jmp    4e9c4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2354>
   4f5c9:	b0 01                	mov    $0x1,%al
   4f5cb:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f5d1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f61>
			4f5cd: R_X86_64_PC32	.bss+0x1fc
   4f5d1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f5d2: R_X86_64_32	.bss+0x208
   4f5d6:	e8 00 00 00 00       	call   4f5db <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f6b>
			4f5d7: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f5db:	41 89 e8             	mov    %ebp,%r8d
   4f5de:	44 89 e1             	mov    %r12d,%ecx
   4f5e1:	4c 89 ea             	mov    %r13,%rdx
   4f5e4:	4c 89 f6             	mov    %r14,%rsi
   4f5e7:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f5ec:	e9 02 d6 ff ff       	jmp    4cbf3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x583>
   4f5f1:	b0 01                	mov    $0x1,%al
   4f5f3:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f5f9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f89>
			4f5f5: R_X86_64_PC32	.bss+0x1fc
   4f5f9:	bf 00 00 00 00       	mov    $0x0,%edi
			4f5fa: R_X86_64_32	.bss+0x208
   4f5fe:	e8 00 00 00 00       	call   4f603 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f93>
			4f5ff: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f603:	41 89 e8             	mov    %ebp,%r8d
   4f606:	44 89 e1             	mov    %r12d,%ecx
   4f609:	4c 89 ea             	mov    %r13,%rdx
   4f60c:	4c 89 f6             	mov    %r14,%rsi
   4f60f:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f614:	e9 c6 d0 ff ff       	jmp    4c6df <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6f>
   4f619:	b0 01                	mov    $0x1,%al
   4f61b:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f621 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fb1>
			4f61d: R_X86_64_PC32	.bss+0x1fc
   4f621:	bf 00 00 00 00       	mov    $0x0,%edi
			4f622: R_X86_64_32	.bss+0x208
   4f626:	e8 00 00 00 00       	call   4f62b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fbb>
			4f627: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f62b:	41 89 e8             	mov    %ebp,%r8d
   4f62e:	44 89 e1             	mov    %r12d,%ecx
   4f631:	4c 89 ea             	mov    %r13,%rdx
   4f634:	4c 89 f6             	mov    %r14,%rsi
   4f637:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f63c:	e9 2d d8 ff ff       	jmp    4ce6e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7fe>
   4f641:	b0 01                	mov    $0x1,%al
   4f643:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f649 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fd9>
			4f645: R_X86_64_PC32	.bss+0x1fc
   4f649:	bf 00 00 00 00       	mov    $0x0,%edi
			4f64a: R_X86_64_32	.bss+0x208
   4f64e:	e8 00 00 00 00       	call   4f653 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fe3>
			4f64f: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f653:	41 89 e8             	mov    %ebp,%r8d
   4f656:	44 89 e1             	mov    %r12d,%ecx
   4f659:	4c 89 ea             	mov    %r13,%rdx
   4f65c:	4c 89 f6             	mov    %r14,%rsi
   4f65f:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f664:	e9 80 da ff ff       	jmp    4d0e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa79>
   4f669:	b0 01                	mov    $0x1,%al
   4f66b:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f671 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3001>
			4f66d: R_X86_64_PC32	.bss+0x1fc
   4f671:	bf 00 00 00 00       	mov    $0x0,%edi
			4f672: R_X86_64_32	.bss+0x208
   4f676:	e8 00 00 00 00       	call   4f67b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x300b>
			4f677: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f67b:	41 89 e8             	mov    %ebp,%r8d
   4f67e:	44 89 e1             	mov    %r12d,%ecx
   4f681:	4c 89 ea             	mov    %r13,%rdx
   4f684:	4c 89 f6             	mov    %r14,%rsi
   4f687:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f68c:	e9 d6 dc ff ff       	jmp    4d367 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcf7>
   4f691:	b0 01                	mov    $0x1,%al
   4f693:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f699 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3029>
			4f695: R_X86_64_PC32	.bss+0x1fc
   4f699:	bf 00 00 00 00       	mov    $0x0,%edi
			4f69a: R_X86_64_32	.bss+0x208
   4f69e:	e8 00 00 00 00       	call   4f6a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3033>
			4f69f: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f6a3:	41 89 e8             	mov    %ebp,%r8d
   4f6a6:	44 89 e1             	mov    %r12d,%ecx
   4f6a9:	4c 89 ea             	mov    %r13,%rdx
   4f6ac:	4c 89 f6             	mov    %r14,%rsi
   4f6af:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f6b4:	e9 10 df ff ff       	jmp    4d5c9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf59>
   4f6b9:	b0 01                	mov    $0x1,%al
   4f6bb:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f6c1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3051>
			4f6bd: R_X86_64_PC32	.bss+0x1fc
   4f6c1:	bf 00 00 00 00       	mov    $0x0,%edi
			4f6c2: R_X86_64_32	.bss+0x208
   4f6c6:	e8 00 00 00 00       	call   4f6cb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x305b>
			4f6c7: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f6cb:	41 89 e8             	mov    %ebp,%r8d
   4f6ce:	44 89 f9             	mov    %r15d,%ecx
   4f6d1:	4c 89 e2             	mov    %r12,%rdx
   4f6d4:	4c 89 ee             	mov    %r13,%rsi
   4f6d7:	4c 89 f7             	mov    %r14,%rdi
   4f6da:	e9 96 d2 ff ff       	jmp    4c975 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x305>
   4f6df:	31 c0                	xor    %eax,%eax
   4f6e1:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4f6e7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3077>
			4f6e3: R_X86_64_PC32	.bss+0x194c
   4f6e7:	bf 00 00 00 00       	mov    $0x0,%edi
			4f6e8: R_X86_64_32	.bss+0x1958
   4f6ec:	e8 00 00 00 00       	call   4f6f1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3081>
			4f6ed: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f6f1:	41 89 e8             	mov    %ebp,%r8d
   4f6f4:	44 89 e1             	mov    %r12d,%ecx
   4f6f7:	4c 89 ea             	mov    %r13,%rdx
   4f6fa:	4c 89 f6             	mov    %r14,%rsi
   4f6fd:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f702:	e9 8d d2 ff ff       	jmp    4c994 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x324>
   4f707:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f70c:	bf 00 00 00 00       	mov    $0x0,%edi
			4f70d: R_X86_64_32	.bss+0x208
   4f711:	44 89 c5             	mov    %r8d,%ebp
   4f714:	41 89 cc             	mov    %ecx,%r12d
   4f717:	49 89 d5             	mov    %rdx,%r13
   4f71a:	49 89 f6             	mov    %rsi,%r14
   4f71d:	e8 00 00 00 00       	call   4f722 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30b2>
			4f71e: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f722:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f727:	4c 89 f6             	mov    %r14,%rsi
   4f72a:	4c 89 ea             	mov    %r13,%rdx
   4f72d:	44 89 e1             	mov    %r12d,%ecx
   4f730:	41 89 e8             	mov    %ebp,%r8d
   4f733:	85 c0                	test   %eax,%eax
   4f735:	0f 84 62 f5 ff ff    	je     4ec9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x262d>
   4f73b:	bf 00 00 00 00       	mov    $0x0,%edi
			4f73c: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f740:	e8 00 00 00 00       	call   4f745 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30d5>
			4f741: R_X86_64_PLT32	getenv-0x4
   4f745:	48 85 c0             	test   %rax,%rax
   4f748:	74 3f                	je     4f789 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3119>
   4f74a:	48 89 c7             	mov    %rax,%rdi
   4f74d:	31 f6                	xor    %esi,%esi
   4f74f:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f754:	e8 00 00 00 00       	call   4f759 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30e9>
			4f755: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f759:	85 c0                	test   %eax,%eax
   4f75b:	0f 95 c0             	setne  %al
   4f75e:	eb 2b                	jmp    4f78b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x311b>
   4f760:	b8 20 00 00 00       	mov    $0x20,%eax
   4f765:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4f76b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30fb>
			4f767: R_X86_64_PC32	.bss+0x4cc
   4f76b:	bf 00 00 00 00       	mov    $0x0,%edi
			4f76c: R_X86_64_32	.bss+0x4d8
   4f770:	e8 00 00 00 00       	call   4f775 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3105>
			4f771: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f775:	41 89 e8             	mov    %ebp,%r8d
   4f778:	44 89 f9             	mov    %r15d,%ecx
   4f77b:	4c 89 e2             	mov    %r12,%rdx
   4f77e:	4c 89 ee             	mov    %r13,%rsi
   4f781:	4c 89 f7             	mov    %r14,%rdi
   4f784:	e9 e0 f0 ff ff       	jmp    4e869 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21f9>
   4f789:	b0 01                	mov    $0x1,%al
   4f78b:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f791 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3121>
			4f78d: R_X86_64_PC32	.bss+0x1fc
   4f791:	bf 00 00 00 00       	mov    $0x0,%edi
			4f792: R_X86_64_32	.bss+0x208
   4f796:	e8 00 00 00 00       	call   4f79b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x312b>
			4f797: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f79b:	41 89 e8             	mov    %ebp,%r8d
   4f79e:	44 89 e1             	mov    %r12d,%ecx
   4f7a1:	4c 89 ea             	mov    %r13,%rdx
   4f7a4:	4c 89 f6             	mov    %r14,%rsi
   4f7a7:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f7ac:	e9 ec f4 ff ff       	jmp    4ec9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x262d>
   4f7b1:	48 89 c3             	mov    %rax,%rbx
   4f7b4:	bf 00 00 00 00       	mov    $0x0,%edi
			4f7b5: R_X86_64_32	.bss+0x1f68
   4f7b9:	e8 00 00 00 00       	call   4f7be <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x314e>
			4f7ba: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f7be:	e9 28 01 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f7c3:	48 89 c3             	mov    %rax,%rbx
   4f7c6:	bf 00 00 00 00       	mov    $0x0,%edi
			4f7c7: R_X86_64_32	.bss+0x1fc8
   4f7cb:	e8 00 00 00 00       	call   4f7d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3160>
			4f7cc: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f7d0:	e9 16 01 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f7d5:	48 89 c3             	mov    %rax,%rbx
   4f7d8:	bf 00 00 00 00       	mov    $0x0,%edi
			4f7d9: R_X86_64_32	.bss+0x1fa8
   4f7dd:	e8 00 00 00 00       	call   4f7e2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3172>
			4f7de: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f7e2:	e9 04 01 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f7e7:	48 89 c3             	mov    %rax,%rbx
   4f7ea:	bf 00 00 00 00       	mov    $0x0,%edi
			4f7eb: R_X86_64_32	.bss+0x1f28
   4f7ef:	e8 00 00 00 00       	call   4f7f4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3184>
			4f7f0: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f7f4:	e9 f2 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f7f9:	48 89 c3             	mov    %rax,%rbx
   4f7fc:	bf 00 00 00 00       	mov    $0x0,%edi
			4f7fd: R_X86_64_32	.bss+0x1ee8
   4f801:	e8 00 00 00 00       	call   4f806 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3196>
			4f802: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f806:	e9 e0 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f80b:	48 89 c3             	mov    %rax,%rbx
   4f80e:	bf 00 00 00 00       	mov    $0x0,%edi
			4f80f: R_X86_64_32	.bss+0x1ec8
   4f813:	e8 00 00 00 00       	call   4f818 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31a8>
			4f814: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f818:	e9 ce 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f81d:	48 89 c3             	mov    %rax,%rbx
   4f820:	bf 00 00 00 00       	mov    $0x0,%edi
			4f821: R_X86_64_32	.bss+0x1f08
   4f825:	e8 00 00 00 00       	call   4f82a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31ba>
			4f826: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f82a:	e9 bc 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f82f:	48 89 c3             	mov    %rax,%rbx
   4f832:	bf 00 00 00 00       	mov    $0x0,%edi
			4f833: R_X86_64_32	.bss+0x1f58
   4f837:	e8 00 00 00 00       	call   4f83c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31cc>
			4f838: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f83c:	e9 aa 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f841:	48 89 c3             	mov    %rax,%rbx
   4f844:	bf 00 00 00 00       	mov    $0x0,%edi
			4f845: R_X86_64_32	.bss+0x1fb8
   4f849:	e8 00 00 00 00       	call   4f84e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31de>
			4f84a: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f84e:	e9 98 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f853:	48 89 c3             	mov    %rax,%rbx
   4f856:	bf 00 00 00 00       	mov    $0x0,%edi
			4f857: R_X86_64_32	.bss+0x1f98
   4f85b:	e8 00 00 00 00       	call   4f860 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x31f0>
			4f85c: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f860:	e9 86 00 00 00       	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f865:	48 89 c3             	mov    %rax,%rbx
   4f868:	bf 00 00 00 00       	mov    $0x0,%edi
			4f869: R_X86_64_32	.bss+0x1f18
   4f86d:	e8 00 00 00 00       	call   4f872 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3202>
			4f86e: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f872:	eb 77                	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f874:	48 89 c3             	mov    %rax,%rbx
   4f877:	bf 00 00 00 00       	mov    $0x0,%edi
			4f878: R_X86_64_32	.bss+0x1ed8
   4f87c:	e8 00 00 00 00       	call   4f881 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3211>
			4f87d: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f881:	eb 68                	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f883:	48 89 c3             	mov    %rax,%rbx
   4f886:	bf 00 00 00 00       	mov    $0x0,%edi
			4f887: R_X86_64_32	.bss+0x1eb8
   4f88b:	e8 00 00 00 00       	call   4f890 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3220>
			4f88c: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f890:	eb 59                	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f892:	48 89 c3             	mov    %rax,%rbx
   4f895:	bf 00 00 00 00       	mov    $0x0,%edi
			4f896: R_X86_64_32	.bss+0x1ef8
   4f89a:	e8 00 00 00 00       	call   4f89f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x322f>
			4f89b: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f89f:	eb 4a                	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f8a1:	48 89 c3             	mov    %rax,%rbx
   4f8a4:	bf 00 00 00 00       	mov    $0x0,%edi
			4f8a5: R_X86_64_32	.bss+0x1fd8
   4f8a9:	e8 00 00 00 00       	call   4f8ae <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x323e>
			4f8aa: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f8ae:	eb 3b                	jmp    4f8eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x327b>
   4f8b0:	eb 36                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8b2:	eb 34                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8b4:	eb 32                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8b6:	eb 30                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8b8:	eb 2e                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8ba:	eb 2c                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8bc:	eb 2a                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8be:	eb 4a                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8c0:	eb 26                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8c2:	eb 46                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8c4:	eb 44                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8c6:	eb 42                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8c8:	eb 40                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8ca:	eb 3e                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8cc:	eb 3c                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8ce:	eb 18                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8d0:	eb 16                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8d2:	eb 14                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8d4:	eb 12                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8d6:	eb 10                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8d8:	eb 0e                	jmp    4f8e8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3278>
   4f8da:	eb 2e                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8dc:	eb 2c                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8de:	eb 2a                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8e0:	eb 28                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8e2:	eb 26                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8e4:	eb 24                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8e6:	eb 22                	jmp    4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
   4f8e8:	48 89 c3             	mov    %rax,%rbx
   4f8eb:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4f8f0:	e8 00 00 00 00       	call   4f8f5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3285>
			4f8f1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f8f5:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4f8fc:	00 
   4f8fd:	e8 00 00 00 00       	call   4f902 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3292>
			4f8fe: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f902:	48 89 df             	mov    %rbx,%rdi
   4f905:	e8 00 00 00 00       	call   4f90a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x329a>
			4f906: R_X86_64_PLT32	_Unwind_Resume-0x4
   4f90a:	48 89 c3             	mov    %rax,%rbx
   4f90d:	48 8d bc 24 88 00 00 	lea    0x88(%rsp),%rdi
   4f914:	00 
   4f915:	e8 00 00 00 00       	call   4f91a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x32aa>
			4f916: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f91a:	48 89 df             	mov    %rbx,%rdi
   4f91d:	e8 00 00 00 00       	call   4f922 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x32b2>
			4f91e: R_X86_64_PLT32	_Unwind_Resume-0x4

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
