
/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q6-a4-s104-compiled-audit/public-native-archive-member.o:     file format elf64-x86-64


Disassembly of section .text:

000000000004c670 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE>:
   4c670:	55                   	push   %rbp
   4c671:	41 57                	push   %r15
   4c673:	41 56                	push   %r14
   4c675:	41 55                	push   %r13
   4c677:	41 54                	push   %r12
   4c679:	53                   	push   %rbx
   4c67a:	48 81 ec 78 04 00 00 	sub    $0x478,%rsp
   4c681:	4c 89 cb             	mov    %r9,%rbx
   4c684:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c68b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b>	4c687: R_X86_64_PC32	.bss+0x1944
   4c68b:	84 c0                	test   %al,%al
   4c68d:	0f 84 9a 26 00 00    	je     4ed2d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26bd>
   4c693:	8b 2d 00 00 00 00    	mov    0x0(%rip),%ebp        # 4c699 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29>	4c695: R_X86_64_PC32	.bss+0x193c
   4c699:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c6a0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30>	4c69c: R_X86_64_PC32	.bss+0x1954
   4c6a0:	84 c0                	test   %al,%al
   4c6a2:	0f 84 d5 26 00 00    	je     4ed7d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x270d>
   4c6a8:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 4c6ae <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3e>	4c6aa: R_X86_64_PC32	.bss+0x194c
   4c6ae:	85 c0                	test   %eax,%eax
   4c6b0:	0f 8e 95 02 00 00    	jle    4c94b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2db>
   4c6b6:	83 c0 fe             	add    $0xfffffffe,%eax
   4c6b9:	83 f8 04             	cmp    $0x4,%eax
   4c6bc:	0f 87 89 02 00 00    	ja     4c94b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2db>
   4c6c2:	4d 63 f8             	movslq %r8d,%r15
   4c6c5:	49 83 c7 03          	add    $0x3,%r15
   4c6c9:	ff 24 c5 00 00 00 00 	jmp    *0x0(,%rax,8)	4c6cc: R_X86_64_32S	.rodata+0x320
   4c6d0:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c6d7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x67>	4c6d3: R_X86_64_PC32	.bss+0x204
   4c6d7:	84 c0                	test   %al,%al
   4c6d9:	0f 84 a3 27 00 00    	je     4ee82 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2812>
   4c6df:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4c6e6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x76>	4c6e1: R_X86_64_PC32	.bss+0x1fb
   4c6e6:	0f 84 bb 15 00 00    	je     4dca7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1637>
   4c6ec:	0f 57 c0             	xorps  %xmm0,%xmm0
   4c6ef:	0f 29 84 24 50 03 00 00 	movaps %xmm0,0x350(%rsp)
   4c6f7:	48 c7 84 24 60 03 00 00 00 00 00 00 	movq   $0x0,0x360(%rsp)
   4c703:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4c708:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4c70d:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4c712:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4c719:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4c721:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4c728 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xb8>	4c724: R_X86_64_PC32	.rodata+0x78224
   4c728:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4c730:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4c737 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xc7>	4c733: R_X86_64_PC32	.rodata+0x78214
   4c737:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4c73f:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4c747:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4c74f:	4c 89 f7             	mov    %r14,%rdi
   4c752:	e8 00 00 00 00       	call   4c757 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe7>	4c753: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c757:	4c 89 f7             	mov    %r14,%rdi
   4c75a:	e8 00 00 00 00       	call   4c75f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xef>	4c75b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4c75f:	49 89 c6             	mov    %rax,%r14
   4c762:	49 c1 e7 03          	shl    $0x3,%r15
   4c766:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4c770:	4c 21 f8             	and    %r15,%rax
   4c773:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4c77f:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4c78b:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4c793:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4c79f:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4c7ab:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4c7b7:	0f 28 84 24 50 03 00 00 	movaps 0x350(%rsp),%xmm0
   4c7bf:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4c7c7:	48 8b 84 24 60 03 00 00 	mov    0x360(%rsp),%rax
   4c7cf:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4c7d7:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4c7df:	4c 89 ff             	mov    %r15,%rdi
   4c7e2:	e8 00 00 00 00       	call   4c7e7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x177>	4c7e3: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4c7e7:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4c7ef:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4c7f7:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4c7ff:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4c807:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4c80f:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4c81b:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c820:	4c 89 f6             	mov    %r14,%rsi
   4c823:	e8 00 00 00 00       	call   4c828 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b8>	4c824: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c828:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c82f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1bf>	4c82b: R_X86_64_PC32	.bss+0x1eb4
   4c82f:	84 c0                	test   %al,%al
   4c831:	0f 84 eb 28 00 00    	je     4f122 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ab2>
   4c837:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4c83e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1ce>	4c83a: R_X86_64_PC32	.bss+0x1eac
   4c83e:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4c846:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4c84e:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4c856:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4c85e:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4c866:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4c86e:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4c876:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4c87e:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4c886:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4c88e:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4c896:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4c89e:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4c8a6:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4c8ae: R_X86_64_32S	.rodata+0x471c8
   4c8b2:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4c8b7:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4c8bf:	0f 57 c0             	xorps  %xmm0,%xmm0
   4c8c2:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4c8ca:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c8cf:	e8 00 00 00 00       	call   4c8d4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x264>	4c8d0: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4c8d4:	31 c9                	xor    %ecx,%ecx
   4c8d6:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4c8db:	0f 94 c1             	sete   %cl
   4c8de:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4c8e6:	0f 11 04 24          	movups %xmm0,(%rsp)
   4c8ea:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4c8ee:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4c8f3:	48 8d bc 24 48 04 00 00 	lea    0x448(%rsp),%rdi
   4c8fb:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4c903:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4c90b:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4c913:	48 89 de             	mov    %rbx,%rsi
   4c916:	4d 89 f0             	mov    %r14,%r8
   4c919:	e8 00 00 00 00       	call   4c91e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ae>	4c91a: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4c91e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4c923:	e8 00 00 00 00       	call   4c928 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b8>	4c924: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4c928:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4c930:	e8 00 00 00 00       	call   4c935 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c5>	4c931: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4c935:	48 8b 9c 24 50 04 00 00 	mov    0x450(%rsp),%rbx
   4c93d:	48 85 db             	test   %rbx,%rbx
   4c940:	0f 85 70 21 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4c946:	e9 c6 21 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4c94b:	83 fd 04             	cmp    $0x4,%ebp
   4c94e:	0f 84 4a 0c 00 00    	je     4d59e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf2e>
   4c954:	83 fd 02             	cmp    $0x2,%ebp
   4c957:	0f 84 df 09 00 00    	je     4d33c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xccc>
   4c95d:	83 fd 01             	cmp    $0x1,%ebp
   4c960:	0f 85 99 0e 00 00    	jne    4d7ff <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x118f>
   4c966:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4c96d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fd>	4c969: R_X86_64_PC32	.bss+0x204
   4c96d:	84 c0                	test   %al,%al
   4c96f:	0f 84 ed 26 00 00    	je     4f062 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29f2>
   4c975:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4c97c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30c>	4c977: R_X86_64_PC32	.bss+0x1fb
   4c97c:	0f 84 bf 1e 00 00    	je     4e841 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21d1>
   4c982:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4c987:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4c98c:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4c991:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4c998:	44 89 c5             	mov    %r8d,%ebp
   4c99b:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4c9a3:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4c9aa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x33a>	4c9a6: R_X86_64_PC32	.rodata+0x78224
   4c9aa:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4c9b2:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4c9b9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x349>	4c9b5: R_X86_64_PC32	.rodata+0x78214
   4c9b9:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4c9c1:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4c9c9:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4c9d1:	4c 89 f7             	mov    %r14,%rdi
   4c9d4:	e8 00 00 00 00       	call   4c9d9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x369>	4c9d5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4c9d9:	4c 89 f7             	mov    %r14,%rdi
   4c9dc:	e8 00 00 00 00       	call   4c9e1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x371>	4c9dd: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4c9e1:	49 89 c6             	mov    %rax,%r14
   4c9e4:	48 63 c5             	movslq %ebp,%rax
   4c9e7:	48 8d 04 c5 18 00 00 00 	lea    0x18(,%rax,8),%rax
   4c9ef:	48 b9 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rcx
   4c9f9:	48 21 c1             	and    %rax,%rcx
   4c9fc:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4ca08:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4ca14:	48 89 8c 24 d8 00 00 00 	mov    %rcx,0xd8(%rsp)
   4ca1c:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4ca28:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4ca34:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4ca40:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ca43:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4ca4b:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4ca57:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4ca5f:	4c 89 ff             	mov    %r15,%rdi
   4ca62:	e8 00 00 00 00       	call   4ca67 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3f7>	4ca63: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4ca67:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4ca6f:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4ca77:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4ca7f:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4ca87:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4ca8f:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4ca9b:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4caa0:	4c 89 f6             	mov    %r14,%rsi
   4caa3:	e8 00 00 00 00       	call   4caa8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x438>	4caa4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4caa8:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4caaf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x43f>	4caab: R_X86_64_PC32	.bss+0x1f34
   4caaf:	84 c0                	test   %al,%al
   4cab1:	0f 84 33 27 00 00    	je     4f1ea <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b7a>
   4cab7:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cabe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x44e>	4caba: R_X86_64_PC32	.bss+0x1f2c
   4cabe:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4cac6:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4cace:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4cad6:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4cade:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4cae6:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4caee:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4caf6:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4cafe:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4cb06:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4cb0e:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4cb16:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4cb1e:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4cb26:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4cb2e: R_X86_64_32S	.rodata+0x48208
   4cb32:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4cb37:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4cb3f:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cb42:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4cb4a:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cb4f:	e8 00 00 00 00       	call   4cb54 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x4e4>	4cb50: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cb54:	31 c9                	xor    %ecx,%ecx
   4cb56:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4cb5b:	0f 94 c1             	sete   %cl
   4cb5e:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4cb66:	0f 11 04 24          	movups %xmm0,(%rsp)
   4cb6a:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4cb6e:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4cb73:	48 8d bc 24 c8 03 00 00 	lea    0x3c8(%rsp),%rdi
   4cb7b:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4cb83:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4cb8b:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4cb93:	48 89 de             	mov    %rbx,%rsi
   4cb96:	4d 89 f0             	mov    %r14,%r8
   4cb99:	e8 00 00 00 00       	call   4cb9e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x52e>	4cb9a: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4cb9e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cba3:	e8 00 00 00 00       	call   4cba8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x538>	4cba4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4cba8:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4cbb0:	e8 00 00 00 00       	call   4cbb5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x545>	4cbb1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4cbb5:	48 8b 9c 24 d0 03 00 00 	mov    0x3d0(%rsp),%rbx
   4cbbd:	48 85 db             	test   %rbx,%rbx
   4cbc0:	0f 85 f0 1e 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4cbc6:	e9 46 1f 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4cbcb:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cbd2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x562>	4cbce: R_X86_64_PC32	.bss+0x204
   4cbd2:	84 c0                	test   %al,%al
   4cbd4:	0f 84 48 22 00 00    	je     4ee22 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x27b2>
   4cbda:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4cbe1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x571>	4cbdc: R_X86_64_PC32	.bss+0x1fb
   4cbe1:	0f 84 61 0e 00 00    	je     4da48 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13d8>
   4cbe7:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cbea:	0f 29 84 24 d0 02 00 00 	movaps %xmm0,0x2d0(%rsp)
   4cbf2:	48 c7 84 24 e0 02 00 00 00 00 00 00 	movq   $0x0,0x2e0(%rsp)
   4cbfe:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4cc03:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4cc08:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4cc0d:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4cc14:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4cc1c:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4cc23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5b3>	4cc1f: R_X86_64_PC32	.rodata+0x78224
   4cc23:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4cc2b:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4cc32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5c2>	4cc2e: R_X86_64_PC32	.rodata+0x78214
   4cc32:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4cc3a:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4cc42:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4cc4a:	4c 89 f7             	mov    %r14,%rdi
   4cc4d:	e8 00 00 00 00       	call   4cc52 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5e2>	4cc4e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cc52:	4c 89 f7             	mov    %r14,%rdi
   4cc55:	e8 00 00 00 00       	call   4cc5a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x5ea>	4cc56: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cc5a:	49 89 c6             	mov    %rax,%r14
   4cc5d:	49 c1 e7 03          	shl    $0x3,%r15
   4cc61:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4cc6b:	4c 21 f8             	and    %r15,%rax
   4cc6e:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4cc7a:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4cc86:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4cc8e:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4cc9a:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4cca6:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4ccb2:	0f 28 84 24 d0 02 00 00 	movaps 0x2d0(%rsp),%xmm0
   4ccba:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4ccc2:	48 8b 84 24 e0 02 00 00 	mov    0x2e0(%rsp),%rax
   4ccca:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4ccd2:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4ccda:	4c 89 ff             	mov    %r15,%rdi
   4ccdd:	e8 00 00 00 00       	call   4cce2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x672>	4ccde: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4cce2:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4ccea:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4ccf2:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4ccfa:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4cd02:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4cd0a:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4cd16:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cd1b:	4c 89 f6             	mov    %r14,%rsi
   4cd1e:	e8 00 00 00 00       	call   4cd23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6b3>	4cd1f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cd23:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cd2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6ba>	4cd26: R_X86_64_PC32	.bss+0x1ef4
   4cd2a:	84 c0                	test   %al,%al
   4cd2c:	0f 84 be 23 00 00    	je     4f0f0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a80>
   4cd32:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cd39 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6c9>	4cd35: R_X86_64_PC32	.bss+0x1eec
   4cd39:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4cd41:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4cd49:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4cd51:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4cd59:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4cd61:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4cd69:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4cd71:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4cd79:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4cd81:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4cd89:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4cd91:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4cd99:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4cda1:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4cda9: R_X86_64_32S	.rodata+0x479e8
   4cdad:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4cdb2:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4cdba:	0f 57 c0             	xorps  %xmm0,%xmm0
   4cdbd:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4cdc5:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cdca:	e8 00 00 00 00       	call   4cdcf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x75f>	4cdcb: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4cdcf:	31 c9                	xor    %ecx,%ecx
   4cdd1:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4cdd6:	0f 94 c1             	sete   %cl
   4cdd9:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4cde1:	0f 11 04 24          	movups %xmm0,(%rsp)
   4cde5:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4cde9:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4cdee:	48 8d bc 24 08 04 00 00 	lea    0x408(%rsp),%rdi
   4cdf6:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4cdfe:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4ce06:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4ce0e:	48 89 de             	mov    %rbx,%rsi
   4ce11:	4d 89 f0             	mov    %r14,%r8
   4ce14:	e8 00 00 00 00       	call   4ce19 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7a9>	4ce15: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4ce19:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ce1e:	e8 00 00 00 00       	call   4ce23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7b3>	4ce1f: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4ce23:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4ce2b:	e8 00 00 00 00       	call   4ce30 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7c0>	4ce2c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4ce30:	48 8b 9c 24 10 04 00 00 	mov    0x410(%rsp),%rbx
   4ce38:	48 85 db             	test   %rbx,%rbx
   4ce3b:	0f 85 75 1c 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4ce41:	e9 cb 1c 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4ce46:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4ce4d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7dd>	4ce49: R_X86_64_PC32	.bss+0x204
   4ce4d:	84 c0                	test   %al,%al
   4ce4f:	0f 84 8d 20 00 00    	je     4eee2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2872>
   4ce55:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4ce5c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7ec>	4ce57: R_X86_64_PC32	.bss+0x1fb
   4ce5c:	0f 84 a4 10 00 00    	je     4df06 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1896>
   4ce62:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ce65:	0f 29 84 24 10 03 00 00 	movaps %xmm0,0x310(%rsp)
   4ce6d:	48 c7 84 24 20 03 00 00 00 00 00 00 	movq   $0x0,0x320(%rsp)
   4ce79:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4ce7e:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4ce83:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4ce88:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4ce8f:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4ce97:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4ce9e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x82e>	4ce9a: R_X86_64_PC32	.rodata+0x78224
   4ce9e:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4cea6:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4cead <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x83d>	4cea9: R_X86_64_PC32	.rodata+0x78214
   4cead:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4ceb5:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4cebd:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4cec5:	4c 89 f7             	mov    %r14,%rdi
   4cec8:	e8 00 00 00 00       	call   4cecd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x85d>	4cec9: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cecd:	4c 89 f7             	mov    %r14,%rdi
   4ced0:	e8 00 00 00 00       	call   4ced5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x865>	4ced1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4ced5:	49 89 c6             	mov    %rax,%r14
   4ced8:	49 c1 e7 03          	shl    $0x3,%r15
   4cedc:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4cee6:	4c 21 f8             	and    %r15,%rax
   4cee9:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4cef5:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4cf01:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4cf09:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4cf15:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4cf21:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4cf2d:	0f 28 84 24 10 03 00 00 	movaps 0x310(%rsp),%xmm0
   4cf35:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4cf3d:	48 8b 84 24 20 03 00 00 	mov    0x320(%rsp),%rax
   4cf45:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4cf4d:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4cf55:	4c 89 ff             	mov    %r15,%rdi
   4cf58:	e8 00 00 00 00       	call   4cf5d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x8ed>	4cf59: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4cf5d:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4cf65:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4cf6d:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4cf75:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4cf7d:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4cf85:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4cf91:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4cf96:	4c 89 f6             	mov    %r14,%rsi
   4cf99:	e8 00 00 00 00       	call   4cf9e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x92e>	4cf9a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4cf9e:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4cfa5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x935>	4cfa1: R_X86_64_PC32	.bss+0x1ed4
   4cfa5:	84 c0                	test   %al,%al
   4cfa7:	0f 84 a7 21 00 00    	je     4f154 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ae4>
   4cfad:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4cfb4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x944>	4cfb0: R_X86_64_PC32	.bss+0x1ecc
   4cfb4:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4cfbc:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4cfc4:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4cfcc:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4cfd4:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4cfdc:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4cfe4:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4cfec:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4cff4:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4cffc:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4d004:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4d00c:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4d014:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4d01c:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4d024: R_X86_64_32S	.rodata+0x475d8
   4d028:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d02d:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4d035:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d038:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4d040:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d045:	e8 00 00 00 00       	call   4d04a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x9da>	4d046: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d04a:	31 c9                	xor    %ecx,%ecx
   4d04c:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d051:	0f 94 c1             	sete   %cl
   4d054:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4d05c:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d060:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d064:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d069:	48 8d bc 24 28 04 00 00 	lea    0x428(%rsp),%rdi
   4d071:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4d079:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4d081:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4d089:	48 89 de             	mov    %rbx,%rsi
   4d08c:	4d 89 f0             	mov    %r14,%r8
   4d08f:	e8 00 00 00 00       	call   4d094 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa24>	4d090: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d094:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d099:	e8 00 00 00 00       	call   4d09e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa2e>	4d09a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d09e:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4d0a6:	e8 00 00 00 00       	call   4d0ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa3b>	4d0a7: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d0ab:	48 8b 9c 24 30 04 00 00 	mov    0x430(%rsp),%rbx
   4d0b3:	48 85 db             	test   %rbx,%rbx
   4d0b6:	0f 85 fa 19 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4d0bc:	e9 50 1a 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4d0c1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d0c8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa58>	4d0c4: R_X86_64_PC32	.bss+0x204
   4d0c8:	84 c0                	test   %al,%al
   4d0ca:	0f 84 72 1e 00 00    	je     4ef42 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28d2>
   4d0d0:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d0d7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa67>	4d0d2: R_X86_64_PC32	.bss+0x1fb
   4d0d7:	0f 84 88 10 00 00    	je     4e165 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1af5>
   4d0dd:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d0e0:	0f 29 84 24 90 02 00 00 	movaps %xmm0,0x290(%rsp)
   4d0e8:	48 c7 84 24 a0 02 00 00 00 00 00 00 	movq   $0x0,0x2a0(%rsp)
   4d0f4:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d0f9:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d0fe:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d103:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d10a:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4d112:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d119 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xaa9>	4d115: R_X86_64_PC32	.rodata+0x78224
   4d119:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4d121:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d128 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xab8>	4d124: R_X86_64_PC32	.rodata+0x78214
   4d128:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4d130:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4d138:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4d140:	4c 89 f7             	mov    %r14,%rdi
   4d143:	e8 00 00 00 00       	call   4d148 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xad8>	4d144: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d148:	4c 89 f7             	mov    %r14,%rdi
   4d14b:	e8 00 00 00 00       	call   4d150 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xae0>	4d14c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d150:	49 89 c6             	mov    %rax,%r14
   4d153:	49 c1 e7 03          	shl    $0x3,%r15
   4d157:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4d161:	4c 21 f8             	and    %r15,%rax
   4d164:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4d170:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4d17c:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4d184:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4d190:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4d19c:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4d1a8:	0f 28 84 24 90 02 00 00 	movaps 0x290(%rsp),%xmm0
   4d1b0:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4d1b8:	48 8b 84 24 a0 02 00 00 	mov    0x2a0(%rsp),%rax
   4d1c0:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4d1c8:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4d1d0:	4c 89 ff             	mov    %r15,%rdi
   4d1d3:	e8 00 00 00 00       	call   4d1d8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xb68>	4d1d4: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d1d8:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4d1e0:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4d1e8:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4d1f0:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4d1f8:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4d200:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4d20c:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d211:	4c 89 f6             	mov    %r14,%rsi
   4d214:	e8 00 00 00 00       	call   4d219 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xba9>	4d215: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d219:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d220 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbb0>	4d21c: R_X86_64_PC32	.bss+0x1f14
   4d220:	84 c0                	test   %al,%al
   4d222:	0f 84 5e 1f 00 00    	je     4f186 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b16>
   4d228:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d22f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbbf>	4d22b: R_X86_64_PC32	.bss+0x1f0c
   4d22f:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4d237:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4d23f:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4d247:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4d24f:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4d257:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4d25f:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4d267:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4d26f:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4d277:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4d27f:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4d287:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4d28f:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4d297:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4d29f: R_X86_64_32S	.rodata+0x47df8
   4d2a3:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d2a8:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4d2b0:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d2b3:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4d2bb:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d2c0:	e8 00 00 00 00       	call   4d2c5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xc55>	4d2c1: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d2c5:	31 c9                	xor    %ecx,%ecx
   4d2c7:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d2cc:	0f 94 c1             	sete   %cl
   4d2cf:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4d2d7:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d2db:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d2df:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d2e4:	48 8d bc 24 e8 03 00 00 	lea    0x3e8(%rsp),%rdi
   4d2ec:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4d2f4:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4d2fc:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4d304:	48 89 de             	mov    %rbx,%rsi
   4d307:	4d 89 f0             	mov    %r14,%r8
   4d30a:	e8 00 00 00 00       	call   4d30f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xc9f>	4d30b: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d30f:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d314:	e8 00 00 00 00       	call   4d319 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xca9>	4d315: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d319:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4d321:	e8 00 00 00 00       	call   4d326 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcb6>	4d322: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d326:	48 8b 9c 24 f0 03 00 00 	mov    0x3f0(%rsp),%rbx
   4d32e:	48 85 db             	test   %rbx,%rbx
   4d331:	0f 85 7f 17 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4d337:	e9 d5 17 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4d33c:	4d 63 f8             	movslq %r8d,%r15
   4d33f:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d346 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcd6>	4d342: R_X86_64_PC32	.bss+0x204
   4d346:	84 c0                	test   %al,%al
   4d348:	0f 84 54 1c 00 00    	je     4efa2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2932>
   4d34e:	49 83 c7 07          	add    $0x7,%r15
   4d352:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d359 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xce9>	4d354: R_X86_64_PC32	.bss+0x1fb
   4d359:	0f 84 65 10 00 00    	je     4e3c4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d54>
   4d35f:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d364:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d369:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d36e:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d375:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4d37d:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d384 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd14>	4d380: R_X86_64_PC32	.rodata+0x78224
   4d384:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4d38c:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d393 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd23>	4d38f: R_X86_64_PC32	.rodata+0x78214
   4d393:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4d39b:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4d3a3:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4d3ab:	4c 89 f7             	mov    %r14,%rdi
   4d3ae:	e8 00 00 00 00       	call   4d3b3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd43>	4d3af: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d3b3:	4c 89 f7             	mov    %r14,%rdi
   4d3b6:	e8 00 00 00 00       	call   4d3bb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xd4b>	4d3b7: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d3bb:	49 89 c6             	mov    %rax,%r14
   4d3be:	49 c1 e7 02          	shl    $0x2,%r15
   4d3c2:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4d3cc:	4c 21 f8             	and    %r15,%rax
   4d3cf:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4d3db:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4d3e7:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4d3ef:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4d3fb:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4d407:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4d413:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d416:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4d41e:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4d42a:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4d432:	4c 89 ff             	mov    %r15,%rdi
   4d435:	e8 00 00 00 00       	call   4d43a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xdca>	4d436: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d43a:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4d442:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4d44a:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4d452:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4d45a:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4d462:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4d46e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d473:	4c 89 f6             	mov    %r14,%rsi
   4d476:	e8 00 00 00 00       	call   4d47b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe0b>	4d477: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d47b:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d482 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe12>	4d47e: R_X86_64_PC32	.bss+0x1f74
   4d482:	84 c0                	test   %al,%al
   4d484:	0f 84 2e 1d 00 00    	je     4f1b8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b48>
   4d48a:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d491 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe21>	4d48d: R_X86_64_PC32	.bss+0x1f6c
   4d491:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4d499:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4d4a1:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4d4a9:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4d4b1:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4d4b9:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4d4c1:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4d4c9:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4d4d1:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4d4d9:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4d4e1:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4d4e9:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4d4f1:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4d4f9:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4d501: R_X86_64_32S	.rodata+0x48a28
   4d505:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d50a:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4d512:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d515:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4d51d:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d522:	e8 00 00 00 00       	call   4d527 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xeb7>	4d523: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d527:	31 c9                	xor    %ecx,%ecx
   4d529:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d52e:	0f 94 c1             	sete   %cl
   4d531:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4d539:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d53d:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d541:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d546:	48 8d bc 24 a8 03 00 00 	lea    0x3a8(%rsp),%rdi
   4d54e:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4d556:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4d55e:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4d566:	48 89 de             	mov    %rbx,%rsi
   4d569:	4d 89 f0             	mov    %r14,%r8
   4d56c:	e8 00 00 00 00       	call   4d571 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf01>	4d56d: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d571:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d576:	e8 00 00 00 00       	call   4d57b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf0b>	4d577: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d57b:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4d583:	e8 00 00 00 00       	call   4d588 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf18>	4d584: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d588:	48 8b 9c 24 b0 03 00 00 	mov    0x3b0(%rsp),%rbx
   4d590:	48 85 db             	test   %rbx,%rbx
   4d593:	0f 85 1d 15 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4d599:	e9 73 15 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4d59e:	4d 63 f8             	movslq %r8d,%r15
   4d5a1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d5a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf38>	4d5a4: R_X86_64_PC32	.bss+0x204
   4d5a8:	84 c0                	test   %al,%al
   4d5aa:	0f 84 52 1a 00 00    	je     4f002 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2992>
   4d5b0:	49 83 c7 0f          	add    $0xf,%r15
   4d5b4:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4d5bb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf4b>	4d5b6: R_X86_64_PC32	.bss+0x1fb
   4d5bb:	0f 84 42 10 00 00    	je     4e603 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f93>
   4d5c1:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d5c6:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d5cb:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d5d0:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d5d7:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4d5df:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d5e6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf76>	4d5e2: R_X86_64_PC32	.rodata+0x78224
   4d5e6:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4d5ee:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d5f5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf85>	4d5f1: R_X86_64_PC32	.rodata+0x78214
   4d5f5:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4d5fd:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4d605:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4d60d:	4c 89 f7             	mov    %r14,%rdi
   4d610:	e8 00 00 00 00       	call   4d615 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xfa5>	4d611: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d615:	4c 89 f7             	mov    %r14,%rdi
   4d618:	e8 00 00 00 00       	call   4d61d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xfad>	4d619: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d61d:	49 89 c6             	mov    %rax,%r14
   4d620:	4d 01 ff             	add    %r15,%r15
   4d623:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4d62d:	4c 21 f8             	and    %r15,%rax
   4d630:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4d63c:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4d648:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4d650:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4d65c:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4d668:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4d674:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d677:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4d67f:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4d68b:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4d693:	4c 89 ff             	mov    %r15,%rdi
   4d696:	e8 00 00 00 00       	call   4d69b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x102b>	4d697: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d69b:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4d6a3:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4d6ab:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4d6b3:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4d6bb:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4d6c3:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4d6cf:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d6d4:	4c 89 f6             	mov    %r14,%rsi
   4d6d7:	e8 00 00 00 00       	call   4d6dc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x106c>	4d6d8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d6dc:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d6e3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1073>	4d6df: R_X86_64_PC32	.bss+0x1f94
   4d6e3:	84 c0                	test   %al,%al
   4d6e5:	0f 84 31 1b 00 00    	je     4f21c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bac>
   4d6eb:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d6f2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1082>	4d6ee: R_X86_64_PC32	.bss+0x1f8c
   4d6f2:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4d6fa:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4d702:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4d70a:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4d712:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4d71a:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4d722:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4d72a:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4d732:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4d73a:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4d742:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4d74a:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4d752:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4d75a:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4d762: R_X86_64_32S	.rodata+0x48e38
   4d766:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d76b:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4d773:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d776:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4d77e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d783:	e8 00 00 00 00       	call   4d788 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1118>	4d784: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d788:	31 c9                	xor    %ecx,%ecx
   4d78a:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d78f:	0f 94 c1             	sete   %cl
   4d792:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4d79a:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d79e:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d7a2:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d7a7:	48 8d bc 24 88 03 00 00 	lea    0x388(%rsp),%rdi
   4d7af:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4d7b7:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4d7bf:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4d7c7:	48 89 de             	mov    %rbx,%rsi
   4d7ca:	4d 89 f0             	mov    %r14,%r8
   4d7cd:	e8 00 00 00 00       	call   4d7d2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1162>	4d7ce: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4d7d2:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d7d7:	e8 00 00 00 00       	call   4d7dc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x116c>	4d7d8: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d7dc:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4d7e4:	e8 00 00 00 00       	call   4d7e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1179>	4d7e5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4d7e9:	48 8b 9c 24 90 03 00 00 	mov    0x390(%rsp),%rbx
   4d7f1:	48 85 db             	test   %rbx,%rbx
   4d7f4:	0f 85 bc 12 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4d7fa:	e9 12 13 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4d7ff:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4d804:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4d809:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4d80e:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4d815:	44 89 c5             	mov    %r8d,%ebp
   4d818:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4d820:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4d827 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11b7>	4d823: R_X86_64_PC32	.rodata+0x78224
   4d827:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4d82f:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4d836 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11c6>	4d832: R_X86_64_PC32	.rodata+0x78214
   4d836:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4d83e:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4d846:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4d84e:	4c 89 f7             	mov    %r14,%rdi
   4d851:	e8 00 00 00 00       	call   4d856 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11e6>	4d852: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d856:	4c 89 f7             	mov    %r14,%rdi
   4d859:	e8 00 00 00 00       	call   4d85e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x11ee>	4d85a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d85e:	49 89 c6             	mov    %rax,%r14
   4d861:	48 63 c5             	movslq %ebp,%rax
   4d864:	48 8d 04 c5 18 00 00 00 	lea    0x18(,%rax,8),%rax
   4d86c:	48 b9 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rcx
   4d876:	48 21 c1             	and    %rax,%rcx
   4d879:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4d885:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4d891:	48 89 8c 24 d8 00 00 00 	mov    %rcx,0xd8(%rsp)
   4d899:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4d8a5:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4d8b1:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4d8bd:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d8c0:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4d8c8:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4d8d4:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4d8dc:	4c 89 ff             	mov    %r15,%rdi
   4d8df:	e8 00 00 00 00       	call   4d8e4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1274>	4d8e0: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4d8e4:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4d8ec:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4d8f4:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4d8fc:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4d904:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4d90c:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4d918:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d91d:	4c 89 f6             	mov    %r14,%rsi
   4d920:	e8 00 00 00 00       	call   4d925 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12b5>	4d921: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4d925:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4d92c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12bc>	4d928: R_X86_64_PC32	.bss+0x1fb4
   4d92c:	84 c0                	test   %al,%al
   4d92e:	0f 84 8a 17 00 00    	je     4f0be <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a4e>
   4d934:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4d93b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12cb>	4d937: R_X86_64_PC32	.bss+0x1fac
   4d93b:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4d943:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4d94b:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4d953:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4d95b:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4d963:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4d96b:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4d973:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4d97b:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4d983:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4d98b:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4d993:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4d99b:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4d9a3:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4d9ab: R_X86_64_32S	.rodata+0x49248
   4d9af:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4d9b4:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4d9bc:	0f 57 c0             	xorps  %xmm0,%xmm0
   4d9bf:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4d9c7:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4d9cc:	e8 00 00 00 00       	call   4d9d1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1361>	4d9cd: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4d9d1:	31 c9                	xor    %ecx,%ecx
   4d9d3:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4d9d8:	0f 94 c1             	sete   %cl
   4d9db:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4d9e3:	0f 11 04 24          	movups %xmm0,(%rsp)
   4d9e7:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4d9eb:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4d9f0:	48 8d bc 24 68 03 00 00 	lea    0x368(%rsp),%rdi
   4d9f8:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4da00:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4da08:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4da10:	48 89 de             	mov    %rbx,%rsi
   4da13:	4d 89 f0             	mov    %r14,%r8
   4da16:	e8 00 00 00 00       	call   4da1b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13ab>	4da17: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4da1b:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4da20:	e8 00 00 00 00       	call   4da25 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13b5>	4da21: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4da25:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4da2d:	e8 00 00 00 00       	call   4da32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x13c2>	4da2e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4da32:	48 8b 9c 24 70 03 00 00 	mov    0x370(%rsp),%rbx
   4da3a:	48 85 db             	test   %rbx,%rbx
   4da3d:	0f 85 73 10 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4da43:	e9 c9 10 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4da48:	0f 57 c0             	xorps  %xmm0,%xmm0
   4da4b:	0f 29 84 24 b0 02 00 00 	movaps %xmm0,0x2b0(%rsp)
   4da53:	48 c7 84 24 c0 02 00 00 00 00 00 00 	movq   $0x0,0x2c0(%rsp)
   4da5f:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4da64:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4da69:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4da6e:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4da75:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4da7d:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4da84 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1414>	4da80: R_X86_64_PC32	.rodata+0x78224
   4da84:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4da8c:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4da93 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1423>	4da8f: R_X86_64_PC32	.rodata+0x78214
   4da93:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4da9b:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4daa3:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4daab:	4c 89 f7             	mov    %r14,%rdi
   4daae:	e8 00 00 00 00       	call   4dab3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1443>	4daaf: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4dab3:	4c 89 f7             	mov    %r14,%rdi
   4dab6:	e8 00 00 00 00       	call   4dabb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x144b>	4dab7: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dabb:	49 89 c6             	mov    %rax,%r14
   4dabe:	49 c1 e7 03          	shl    $0x3,%r15
   4dac2:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4dacc:	4c 21 f8             	and    %r15,%rax
   4dacf:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4dadb:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4dae7:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4daef:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4dafb:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4db07:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4db13:	0f 28 84 24 b0 02 00 00 	movaps 0x2b0(%rsp),%xmm0
   4db1b:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4db23:	48 8b 84 24 c0 02 00 00 	mov    0x2c0(%rsp),%rax
   4db2b:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4db33:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4db3b:	4c 89 ff             	mov    %r15,%rdi
   4db3e:	e8 00 00 00 00       	call   4db43 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x14d3>	4db3f: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4db43:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4db4b:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4db53:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4db5b:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4db63:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4db6b:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4db77:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4db7c:	4c 89 f6             	mov    %r14,%rsi
   4db7f:	e8 00 00 00 00       	call   4db84 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1514>	4db80: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4db84:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4db8b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x151b>	4db87: R_X86_64_PC32	.bss+0x1f04
   4db8b:	84 c0                	test   %al,%al
   4db8d:	0f 84 12 17 00 00    	je     4f2a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c35>
   4db93:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4db9a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x152a>	4db96: R_X86_64_PC32	.bss+0x1efc
   4db9a:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4dba2:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4dbaa:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4dbb2:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4dbba:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4dbc2:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4dbca:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4dbd2:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4dbda:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4dbe2:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4dbea:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4dbf2:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4dbfa:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4dc02:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4dc0a: R_X86_64_32S	.rodata+0x47bf0
   4dc0e:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4dc13:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4dc1b:	0f 57 c0             	xorps  %xmm0,%xmm0
   4dc1e:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4dc26:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dc2b:	e8 00 00 00 00       	call   4dc30 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x15c0>	4dc2c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dc30:	31 c9                	xor    %ecx,%ecx
   4dc32:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4dc37:	0f 94 c1             	sete   %cl
   4dc3a:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4dc42:	0f 11 04 24          	movups %xmm0,(%rsp)
   4dc46:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4dc4a:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4dc4f:	48 8d bc 24 f8 03 00 00 	lea    0x3f8(%rsp),%rdi
   4dc57:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4dc5f:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4dc67:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4dc6f:	48 89 de             	mov    %rbx,%rsi
   4dc72:	4d 89 f0             	mov    %r14,%r8
   4dc75:	e8 00 00 00 00       	call   4dc7a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x160a>	4dc76: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4dc7a:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dc7f:	e8 00 00 00 00       	call   4dc84 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1614>	4dc80: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4dc84:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4dc8c:	e8 00 00 00 00       	call   4dc91 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1621>	4dc8d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4dc91:	48 8b 9c 24 00 04 00 00 	mov    0x400(%rsp),%rbx
   4dc99:	48 85 db             	test   %rbx,%rbx
   4dc9c:	0f 85 14 0e 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4dca2:	e9 6a 0e 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4dca7:	0f 57 c0             	xorps  %xmm0,%xmm0
   4dcaa:	0f 29 84 24 30 03 00 00 	movaps %xmm0,0x330(%rsp)
   4dcb2:	48 c7 84 24 40 03 00 00 00 00 00 00 	movq   $0x0,0x340(%rsp)
   4dcbe:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4dcc3:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4dcc8:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4dccd:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4dcd4:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4dcdc:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4dce3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1673>	4dcdf: R_X86_64_PC32	.rodata+0x78224
   4dce3:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4dceb:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4dcf2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1682>	4dcee: R_X86_64_PC32	.rodata+0x78214
   4dcf2:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4dcfa:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4dd02:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4dd0a:	4c 89 f7             	mov    %r14,%rdi
   4dd0d:	e8 00 00 00 00       	call   4dd12 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x16a2>	4dd0e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4dd12:	4c 89 f7             	mov    %r14,%rdi
   4dd15:	e8 00 00 00 00       	call   4dd1a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x16aa>	4dd16: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4dd1a:	49 89 c6             	mov    %rax,%r14
   4dd1d:	49 c1 e7 03          	shl    $0x3,%r15
   4dd21:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4dd2b:	4c 21 f8             	and    %r15,%rax
   4dd2e:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4dd3a:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4dd46:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4dd4e:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4dd5a:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4dd66:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4dd72:	0f 28 84 24 30 03 00 00 	movaps 0x330(%rsp),%xmm0
   4dd7a:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4dd82:	48 8b 84 24 40 03 00 00 	mov    0x340(%rsp),%rax
   4dd8a:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4dd92:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4dd9a:	4c 89 ff             	mov    %r15,%rdi
   4dd9d:	e8 00 00 00 00       	call   4dda2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1732>	4dd9e: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4dda2:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4ddaa:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4ddb2:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4ddba:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4ddc2:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4ddca:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4ddd6:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dddb:	4c 89 f6             	mov    %r14,%rsi
   4ddde:	e8 00 00 00 00       	call   4dde3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1773>	4dddf: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4dde3:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4ddea <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x177a>	4dde6: R_X86_64_PC32	.bss+0x1ec4
   4ddea:	84 c0                	test   %al,%al
   4ddec:	0f 84 e5 14 00 00    	je     4f2d7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c67>
   4ddf2:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4ddf9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1789>	4ddf5: R_X86_64_PC32	.bss+0x1ebc
   4ddf9:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4de01:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4de09:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4de11:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4de19:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4de21:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4de29:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4de31:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4de39:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4de41:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4de49:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4de51:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4de59:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4de61:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4de69: R_X86_64_32S	.rodata+0x473d0
   4de6d:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4de72:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4de7a:	0f 57 c0             	xorps  %xmm0,%xmm0
   4de7d:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4de85:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4de8a:	e8 00 00 00 00       	call   4de8f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x181f>	4de8b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4de8f:	31 c9                	xor    %ecx,%ecx
   4de91:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4de96:	0f 94 c1             	sete   %cl
   4de99:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4dea1:	0f 11 04 24          	movups %xmm0,(%rsp)
   4dea5:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4dea9:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4deae:	48 8d bc 24 38 04 00 00 	lea    0x438(%rsp),%rdi
   4deb6:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4debe:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4dec6:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4dece:	48 89 de             	mov    %rbx,%rsi
   4ded1:	4d 89 f0             	mov    %r14,%r8
   4ded4:	e8 00 00 00 00       	call   4ded9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1869>	4ded5: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4ded9:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4dede:	e8 00 00 00 00       	call   4dee3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1873>	4dedf: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4dee3:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4deeb:	e8 00 00 00 00       	call   4def0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1880>	4deec: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4def0:	48 8b 9c 24 40 04 00 00 	mov    0x440(%rsp),%rbx
   4def8:	48 85 db             	test   %rbx,%rbx
   4defb:	0f 85 b5 0b 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4df01:	e9 0b 0c 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4df06:	0f 57 c0             	xorps  %xmm0,%xmm0
   4df09:	0f 29 84 24 f0 02 00 00 	movaps %xmm0,0x2f0(%rsp)
   4df11:	48 c7 84 24 00 03 00 00 00 00 00 00 	movq   $0x0,0x300(%rsp)
   4df1d:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4df22:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4df27:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4df2c:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4df33:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4df3b:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4df42 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x18d2>	4df3e: R_X86_64_PC32	.rodata+0x78224
   4df42:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4df4a:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4df51 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x18e1>	4df4d: R_X86_64_PC32	.rodata+0x78214
   4df51:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4df59:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4df61:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4df69:	4c 89 f7             	mov    %r14,%rdi
   4df6c:	e8 00 00 00 00       	call   4df71 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1901>	4df6d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4df71:	4c 89 f7             	mov    %r14,%rdi
   4df74:	e8 00 00 00 00       	call   4df79 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1909>	4df75: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4df79:	49 89 c6             	mov    %rax,%r14
   4df7c:	49 c1 e7 03          	shl    $0x3,%r15
   4df80:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4df8a:	4c 21 f8             	and    %r15,%rax
   4df8d:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4df99:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4dfa5:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4dfad:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4dfb9:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4dfc5:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4dfd1:	0f 28 84 24 f0 02 00 00 	movaps 0x2f0(%rsp),%xmm0
   4dfd9:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4dfe1:	48 8b 84 24 00 03 00 00 	mov    0x300(%rsp),%rax
   4dfe9:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4dff1:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4dff9:	4c 89 ff             	mov    %r15,%rdi
   4dffc:	e8 00 00 00 00       	call   4e001 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1991>	4dffd: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e001:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4e009:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4e011:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4e019:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4e021:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4e029:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4e035:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e03a:	4c 89 f6             	mov    %r14,%rsi
   4e03d:	e8 00 00 00 00       	call   4e042 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19d2>	4e03e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e042:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e049 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19d9>	4e045: R_X86_64_PC32	.bss+0x1ee4
   4e049:	84 c0                	test   %al,%al
   4e04b:	0f 84 b8 12 00 00    	je     4f309 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c99>
   4e051:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e058 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19e8>	4e054: R_X86_64_PC32	.bss+0x1edc
   4e058:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4e060:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4e068:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4e070:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4e078:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4e080:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4e088:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4e090:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4e098:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4e0a0:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4e0a8:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4e0b0:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4e0b8:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4e0c0:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4e0c8: R_X86_64_32S	.rodata+0x477e0
   4e0cc:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e0d1:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4e0d9:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e0dc:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4e0e4:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e0e9:	e8 00 00 00 00       	call   4e0ee <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1a7e>	4e0ea: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e0ee:	31 c9                	xor    %ecx,%ecx
   4e0f0:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e0f5:	0f 94 c1             	sete   %cl
   4e0f8:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4e100:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e104:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e108:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e10d:	48 8d bc 24 18 04 00 00 	lea    0x418(%rsp),%rdi
   4e115:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4e11d:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4e125:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4e12d:	48 89 de             	mov    %rbx,%rsi
   4e130:	4d 89 f0             	mov    %r14,%r8
   4e133:	e8 00 00 00 00       	call   4e138 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1ac8>	4e134: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e138:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e13d:	e8 00 00 00 00       	call   4e142 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1ad2>	4e13e: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e142:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4e14a:	e8 00 00 00 00       	call   4e14f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1adf>	4e14b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e14f:	48 8b 9c 24 20 04 00 00 	mov    0x420(%rsp),%rbx
   4e157:	48 85 db             	test   %rbx,%rbx
   4e15a:	0f 85 56 09 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4e160:	e9 ac 09 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4e165:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e168:	0f 29 84 24 70 02 00 00 	movaps %xmm0,0x270(%rsp)
   4e170:	48 c7 84 24 80 02 00 00 00 00 00 00 	movq   $0x0,0x280(%rsp)
   4e17c:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e181:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e186:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e18b:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e192:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4e19a:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e1a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b31>	4e19d: R_X86_64_PC32	.rodata+0x78224
   4e1a1:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4e1a9:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e1b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b40>	4e1ac: R_X86_64_PC32	.rodata+0x78214
   4e1b0:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4e1b8:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4e1c0:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4e1c8:	4c 89 f7             	mov    %r14,%rdi
   4e1cb:	e8 00 00 00 00       	call   4e1d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b60>	4e1cc: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e1d0:	4c 89 f7             	mov    %r14,%rdi
   4e1d3:	e8 00 00 00 00       	call   4e1d8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1b68>	4e1d4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e1d8:	49 89 c6             	mov    %rax,%r14
   4e1db:	49 c1 e7 03          	shl    $0x3,%r15
   4e1df:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4e1e9:	4c 21 f8             	and    %r15,%rax
   4e1ec:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4e1f8:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4e204:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4e20c:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4e218:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4e224:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4e230:	0f 28 84 24 70 02 00 00 	movaps 0x270(%rsp),%xmm0
   4e238:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4e240:	48 8b 84 24 80 02 00 00 	mov    0x280(%rsp),%rax
   4e248:	48 89 84 24 08 01 00 00 	mov    %rax,0x108(%rsp)
   4e250:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4e258:	4c 89 ff             	mov    %r15,%rdi
   4e25b:	e8 00 00 00 00       	call   4e260 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1bf0>	4e25c: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e260:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4e268:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4e270:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4e278:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4e280:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4e288:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4e294:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e299:	4c 89 f6             	mov    %r14,%rsi
   4e29c:	e8 00 00 00 00       	call   4e2a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c31>	4e29d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e2a1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e2a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c38>	4e2a4: R_X86_64_PC32	.bss+0x1f24
   4e2a8:	84 c0                	test   %al,%al
   4e2aa:	0f 84 8b 10 00 00    	je     4f33b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ccb>
   4e2b0:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e2b7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c47>	4e2b3: R_X86_64_PC32	.bss+0x1f1c
   4e2b7:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4e2bf:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4e2c7:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4e2cf:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4e2d7:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4e2df:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4e2e7:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4e2ef:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4e2f7:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4e2ff:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4e307:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4e30f:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4e317:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4e31f:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4e327: R_X86_64_32S	.rodata+0x48000
   4e32b:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e330:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4e338:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e33b:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4e343:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e348:	e8 00 00 00 00       	call   4e34d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1cdd>	4e349: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e34d:	31 c9                	xor    %ecx,%ecx
   4e34f:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e354:	0f 94 c1             	sete   %cl
   4e357:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4e35f:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e363:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e367:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e36c:	48 8d bc 24 d8 03 00 00 	lea    0x3d8(%rsp),%rdi
   4e374:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4e37c:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4e384:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4e38c:	48 89 de             	mov    %rbx,%rsi
   4e38f:	4d 89 f0             	mov    %r14,%r8
   4e392:	e8 00 00 00 00       	call   4e397 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d27>	4e393: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e397:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e39c:	e8 00 00 00 00       	call   4e3a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d31>	4e39d: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e3a1:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4e3a9:	e8 00 00 00 00       	call   4e3ae <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d3e>	4e3aa: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e3ae:	48 8b 9c 24 e0 03 00 00 	mov    0x3e0(%rsp),%rbx
   4e3b6:	48 85 db             	test   %rbx,%rbx
   4e3b9:	0f 85 f7 06 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4e3bf:	e9 4d 07 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4e3c4:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e3c9:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e3ce:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e3d3:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e3da:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4e3e2:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e3e9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d79>	4e3e5: R_X86_64_PC32	.rodata+0x78224
   4e3e9:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4e3f1:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e3f8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1d88>	4e3f4: R_X86_64_PC32	.rodata+0x78214
   4e3f8:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4e400:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4e408:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4e410:	4c 89 f7             	mov    %r14,%rdi
   4e413:	e8 00 00 00 00       	call   4e418 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1da8>	4e414: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e418:	4c 89 f7             	mov    %r14,%rdi
   4e41b:	e8 00 00 00 00       	call   4e420 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1db0>	4e41c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e420:	49 89 c6             	mov    %rax,%r14
   4e423:	49 c1 e7 02          	shl    $0x2,%r15
   4e427:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4e431:	4c 21 f8             	and    %r15,%rax
   4e434:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4e440:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4e44c:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4e454:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4e460:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4e46c:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4e478:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e47b:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4e483:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4e48f:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4e497:	4c 89 ff             	mov    %r15,%rdi
   4e49a:	e8 00 00 00 00       	call   4e49f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e2f>	4e49b: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e49f:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4e4a7:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4e4af:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4e4b7:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4e4bf:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4e4c7:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4e4d3:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e4d8:	4c 89 f6             	mov    %r14,%rsi
   4e4db:	e8 00 00 00 00       	call   4e4e0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e70>	4e4dc: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e4e0:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e4e7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e77>	4e4e3: R_X86_64_PC32	.bss+0x1f84
   4e4e7:	84 c0                	test   %al,%al
   4e4e9:	0f 84 7e 0e 00 00    	je     4f36d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cfd>
   4e4ef:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e4f6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e86>	4e4f2: R_X86_64_PC32	.bss+0x1f7c
   4e4f6:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4e4fe:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4e506:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4e50e:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4e516:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4e51e:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4e526:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4e52e:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4e536:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4e53e:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4e546:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4e54e:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4e556:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4e55e:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4e566: R_X86_64_32S	.rodata+0x48c30
   4e56a:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e56f:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4e577:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e57a:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4e582:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e587:	e8 00 00 00 00       	call   4e58c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f1c>	4e588: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e58c:	31 c9                	xor    %ecx,%ecx
   4e58e:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e593:	0f 94 c1             	sete   %cl
   4e596:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4e59e:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e5a2:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e5a6:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e5ab:	48 8d bc 24 98 03 00 00 	lea    0x398(%rsp),%rdi
   4e5b3:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4e5bb:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4e5c3:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4e5cb:	48 89 de             	mov    %rbx,%rsi
   4e5ce:	4d 89 f0             	mov    %r14,%r8
   4e5d1:	e8 00 00 00 00       	call   4e5d6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f66>	4e5d2: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e5d6:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e5db:	e8 00 00 00 00       	call   4e5e0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f70>	4e5dc: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e5e0:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4e5e8:	e8 00 00 00 00       	call   4e5ed <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1f7d>	4e5e9: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e5ed:	48 8b 9c 24 a0 03 00 00 	mov    0x3a0(%rsp),%rbx
   4e5f5:	48 85 db             	test   %rbx,%rbx
   4e5f8:	0f 85 b8 04 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4e5fe:	e9 0e 05 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4e603:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e608:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e60d:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e612:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e619:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4e621:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e628 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fb8>	4e624: R_X86_64_PC32	.rodata+0x78224
   4e628:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4e630:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e637 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fc7>	4e633: R_X86_64_PC32	.rodata+0x78214
   4e637:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4e63f:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4e647:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4e64f:	4c 89 f7             	mov    %r14,%rdi
   4e652:	e8 00 00 00 00       	call   4e657 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fe7>	4e653: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e657:	4c 89 f7             	mov    %r14,%rdi
   4e65a:	e8 00 00 00 00       	call   4e65f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1fef>	4e65b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e65f:	49 89 c6             	mov    %rax,%r14
   4e662:	4d 01 ff             	add    %r15,%r15
   4e665:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4e66f:	4c 21 f8             	and    %r15,%rax
   4e672:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4e67e:	48 c7 84 24 d0 00 00 00 04 00 00 00 	movq   $0x4,0xd0(%rsp)
   4e68a:	48 89 84 24 d8 00 00 00 	mov    %rax,0xd8(%rsp)
   4e692:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4e69e:	48 c7 84 24 e8 00 00 00 04 00 00 00 	movq   $0x4,0xe8(%rsp)
   4e6aa:	48 c7 84 24 f0 00 00 00 20 00 00 00 	movq   $0x20,0xf0(%rsp)
   4e6b6:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e6b9:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4e6c1:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4e6cd:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4e6d5:	4c 89 ff             	mov    %r15,%rdi
   4e6d8:	e8 00 00 00 00       	call   4e6dd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x206d>	4e6d9: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e6dd:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4e6e5:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4e6ed:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4e6f5:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4e6fd:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4e705:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4e711:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e716:	4c 89 f6             	mov    %r14,%rsi
   4e719:	e8 00 00 00 00       	call   4e71e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20ae>	4e71a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e71e:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e725 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20b5>	4e721: R_X86_64_PC32	.bss+0x1fa4
   4e725:	84 c0                	test   %al,%al
   4e727:	0f 84 72 0c 00 00    	je     4f39f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d2f>
   4e72d:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e734 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20c4>	4e730: R_X86_64_PC32	.bss+0x1f9c
   4e734:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4e73c:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4e744:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4e74c:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4e754:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4e75c:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4e764:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4e76c:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4e774:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4e77c:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4e784:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4e78c:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4e794:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4e79c:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4e7a4: R_X86_64_32S	.rodata+0x49040
   4e7a8:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4e7ad:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4e7b5:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e7b8:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4e7c0:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e7c5:	e8 00 00 00 00       	call   4e7ca <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x215a>	4e7c6: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e7ca:	31 c9                	xor    %ecx,%ecx
   4e7cc:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4e7d1:	0f 94 c1             	sete   %cl
   4e7d4:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4e7dc:	0f 11 04 24          	movups %xmm0,(%rsp)
   4e7e0:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4e7e4:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4e7e9:	48 8d bc 24 78 03 00 00 	lea    0x378(%rsp),%rdi
   4e7f1:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4e7f9:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4e801:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4e809:	48 89 de             	mov    %rbx,%rsi
   4e80c:	4d 89 f0             	mov    %r14,%r8
   4e80f:	e8 00 00 00 00       	call   4e814 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21a4>	4e810: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4e814:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e819:	e8 00 00 00 00       	call   4e81e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21ae>	4e81a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e81e:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4e826:	e8 00 00 00 00       	call   4e82b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21bb>	4e827: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4e82b:	48 8b 9c 24 80 03 00 00 	mov    0x380(%rsp),%rbx
   4e833:	48 85 db             	test   %rbx,%rbx
   4e836:	0f 85 7a 02 00 00    	jne    4eab6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2446>
   4e83c:	e9 d0 02 00 00       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4e841:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e848 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21d8>	4e844: R_X86_64_PC32	.bss+0x4d4
   4e848:	84 c0                	test   %al,%al
   4e84a:	0f 84 fe 09 00 00    	je     4f24e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bde>
   4e850:	8b 05 00 00 00 00    	mov    0x0(%rip),%eax        # 4e856 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21e6>	4e852: R_X86_64_PC32	.bss+0x4cc
   4e856:	4d 63 f8             	movslq %r8d,%r15
   4e859:	83 f8 10             	cmp    $0x10,%eax
   4e85c:	74 1e                	je     4e87c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x220c>
   4e85e:	a8 02                	test   $0x2,%al
   4e860:	0f 84 bd 02 00 00    	je     4eb23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24b3>
   4e866:	41 81 f8 00 04 00 00 	cmp    $0x400,%r8d
   4e86d:	0f 8f b0 02 00 00    	jg     4eb23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24b3>
   4e873:	83 f8 04             	cmp    $0x4,%eax
   4e876:	0f 8d a7 02 00 00    	jge    4eb23 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24b3>
   4e87c:	48 89 7c 24 68       	mov    %rdi,0x68(%rsp)
   4e881:	48 89 74 24 70       	mov    %rsi,0x70(%rsp)
   4e886:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
   4e88b:	89 8c 24 80 00 00 00 	mov    %ecx,0x80(%rsp)
   4e892:	44 89 84 24 84 00 00 00 	mov    %r8d,0x84(%rsp)
   4e89a:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 4e8a1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2231>	4e89d: R_X86_64_PC32	.rodata+0x78224
   4e8a1:	48 89 84 24 20 01 00 00 	mov    %rax,0x120(%rsp)
   4e8a9:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 4e8b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2240>	4e8ac: R_X86_64_PC32	.rodata+0x78214
   4e8b0:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   4e8b8:	4c 8d b4 24 88 00 00 00 	lea    0x88(%rsp),%r14
   4e8c0:	48 8d b4 24 10 01 00 00 	lea    0x110(%rsp),%rsi
   4e8c8:	4c 89 f7             	mov    %r14,%rdi
   4e8cb:	e8 00 00 00 00       	call   4e8d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2260>	4e8cc: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e8d0:	4c 89 f7             	mov    %r14,%rdi
   4e8d3:	e8 00 00 00 00       	call   4e8d8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2268>	4e8d4: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4e8d8:	49 89 c6             	mov    %rax,%r14
   4e8db:	4a 8d 04 7d 0e 00 00 00 	lea    0xe(,%r15,2),%rax
   4e8e3:	48 b9 f0 ff ff ff 0f 00 00 00 	movabs $0xffffffff0,%rcx
   4e8ed:	48 21 c1             	and    %rax,%rcx
   4e8f0:	48 c7 84 24 c8 00 00 00 01 00 00 00 	movq   $0x1,0xc8(%rsp)
   4e8fc:	48 c7 84 24 d0 00 00 00 08 00 00 00 	movq   $0x8,0xd0(%rsp)
   4e908:	48 89 8c 24 d8 00 00 00 	mov    %rcx,0xd8(%rsp)
   4e910:	48 c7 84 24 e0 00 00 00 01 00 00 00 	movq   $0x1,0xe0(%rsp)
   4e91c:	48 c7 84 24 e8 00 00 00 08 00 00 00 	movq   $0x8,0xe8(%rsp)
   4e928:	48 c7 84 24 f0 00 00 00 10 00 00 00 	movq   $0x10,0xf0(%rsp)
   4e934:	0f 57 c0             	xorps  %xmm0,%xmm0
   4e937:	0f 11 84 24 f8 00 00 00 	movups %xmm0,0xf8(%rsp)
   4e93f:	48 c7 84 24 08 01 00 00 00 00 00 00 	movq   $0x0,0x108(%rsp)
   4e94b:	4c 8d bc 24 c8 00 00 00 	lea    0xc8(%rsp),%r15
   4e953:	4c 89 ff             	mov    %r15,%rdi
   4e956:	e8 00 00 00 00       	call   4e95b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x22eb>	4e957: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   4e95b:	48 8d 84 24 e0 00 00 00 	lea    0xe0(%rsp),%rax
   4e963:	48 8d 8c 24 f8 00 00 00 	lea    0xf8(%rsp),%rcx
   4e96b:	4c 89 bc 24 a8 00 00 00 	mov    %r15,0xa8(%rsp)
   4e973:	48 89 84 24 b0 00 00 00 	mov    %rax,0xb0(%rsp)
   4e97b:	48 89 8c 24 b8 00 00 00 	mov    %rcx,0xb8(%rsp)
   4e983:	48 c7 84 24 c0 00 00 00 03 00 00 00 	movq   $0x3,0xc0(%rsp)
   4e98f:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4e994:	4c 89 f6             	mov    %r14,%rsi
   4e997:	e8 00 00 00 00       	call   4e99c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x232c>	4e998: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   4e99c:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4e9a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2333>	4e99f: R_X86_64_PC32	.bss+0x1f44
   4e9a3:	84 c0                	test   %al,%al
   4e9a5:	0f 84 26 0a 00 00    	je     4f3d1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d61>
   4e9ab:	4c 8b 35 00 00 00 00 	mov    0x0(%rip),%r14        # 4e9b2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2342>	4e9ae: R_X86_64_PC32	.bss+0x1f3c
   4e9b2:	c6 84 24 50 01 00 00 00 	movb   $0x0,0x150(%rsp)
   4e9ba:	c6 84 24 5a 01 00 00 00 	movb   $0x0,0x15a(%rsp)
   4e9c2:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   4e9ca:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   4e9d2:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   4e9da:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   4e9e2:	c6 84 24 7c 01 00 00 00 	movb   $0x0,0x17c(%rsp)
   4e9ea:	c6 84 24 84 01 00 00 00 	movb   $0x0,0x184(%rsp)
   4e9f2:	c6 84 24 8c 01 00 00 00 	movb   $0x0,0x18c(%rsp)
   4e9fa:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   4ea02:	c6 84 24 b0 01 00 00 00 	movb   $0x0,0x1b0(%rsp)
   4ea0a:	c6 84 24 d0 01 00 00 00 	movb   $0x0,0x1d0(%rsp)
   4ea12:	c6 84 24 d8 01 00 00 01 	movb   $0x1,0x1d8(%rsp)
   4ea1a:	48 c7 84 24 98 00 00 00 00 00 00 00 	movq   $0x0,0x98(%rsp)	4ea22: R_X86_64_32S	.rodata+0x48410
   4ea26:	48 8d 44 24 68       	lea    0x68(%rsp),%rax
   4ea2b:	48 89 84 24 a0 00 00 00 	mov    %rax,0xa0(%rsp)
   4ea33:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ea36:	0f 29 84 24 30 01 00 00 	movaps %xmm0,0x130(%rsp)
   4ea3e:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ea43:	e8 00 00 00 00       	call   4ea48 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23d8>	4ea44: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   4ea48:	31 c9                	xor    %ecx,%ecx
   4ea4a:	80 7c 24 58 00       	cmpb   $0x0,0x58(%rsp)
   4ea4f:	0f 94 c1             	sete   %cl
   4ea52:	0f 28 84 24 30 01 00 00 	movaps 0x130(%rsp),%xmm0
   4ea5a:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ea5e:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   4ea62:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   4ea67:	48 8d bc 24 b8 03 00 00 	lea    0x3b8(%rsp),%rdi
   4ea6f:	48 8d 94 24 a8 00 00 00 	lea    0xa8(%rsp),%rdx
   4ea77:	48 8d 8c 24 98 00 00 00 	lea    0x98(%rsp),%rcx
   4ea7f:	4c 8d 8c 24 48 01 00 00 	lea    0x148(%rsp),%r9
   4ea87:	48 89 de             	mov    %rbx,%rsi
   4ea8a:	4d 89 f0             	mov    %r14,%r8
   4ea8d:	e8 00 00 00 00       	call   4ea92 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2422>	4ea8e: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   4ea92:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4ea97:	e8 00 00 00 00       	call   4ea9c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x242c>	4ea98: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4ea9c:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4eaa4:	e8 00 00 00 00       	call   4eaa9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2439>	4eaa5: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4eaa9:	48 8b 9c 24 c0 03 00 00 	mov    0x3c0(%rsp),%rbx
   4eab1:	48 85 db             	test   %rbx,%rbx
   4eab4:	74 5b                	je     4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4eab6:	48 8b 43 08          	mov    0x8(%rbx),%rax
   4eaba:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
   4eac4:	48 39 c8             	cmp    %rcx,%rax
   4eac7:	75 1c                	jne    4eae5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2475>
   4eac9:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
   4ead1:	48 8b 03             	mov    (%rbx),%rax
   4ead4:	48 89 df             	mov    %rbx,%rdi
   4ead7:	ff 50 10             	call   *0x10(%rax)
   4eada:	48 8b 03             	mov    (%rbx),%rax
   4eadd:	48 89 df             	mov    %rbx,%rdi
   4eae0:	ff 50 18             	call   *0x18(%rax)
   4eae3:	eb 2c                	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4eae5:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4eaec <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x247c>	4eae7: R_X86_64_PC32	__libc_single_threaded-0x5
   4eaec:	74 10                	je     4eafe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x248e>
   4eaee:	8d 48 ff             	lea    -0x1(%rax),%ecx
   4eaf1:	89 4b 08             	mov    %ecx,0x8(%rbx)
   4eaf4:	83 f8 01             	cmp    $0x1,%eax
   4eaf7:	75 18                	jne    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4eaf9:	e9 22 02 00 00       	jmp    4ed20 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26b0>
   4eafe:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
   4eb03:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
   4eb08:	83 f8 01             	cmp    $0x1,%eax
   4eb0b:	0f 84 0f 02 00 00    	je     4ed20 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26b0>
   4eb11:	48 81 c4 78 04 00 00 	add    $0x478,%rsp
   4eb18:	5b                   	pop    %rbx
   4eb19:	41 5c                	pop    %r12
   4eb1b:	41 5d                	pop    %r13
   4eb1d:	41 5e                	pop    %r14
   4eb1f:	41 5f                	pop    %r15
   4eb21:	5d                   	pop    %rbp
   4eb22:	c3                   	ret
   4eb23:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 4eb2a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24ba>	4eb26: R_X86_64_PC32	.bss+0x204
   4eb2a:	84 c0                	test   %al,%al
   4eb2c:	0f 84 e7 09 00 00    	je     4f519 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ea9>
   4eb32:	4e 8d 0c fd 18 00 00 00 	lea    0x18(,%r15,8),%r9
   4eb3a:	48 b8 e0 ff ff ff 1f 00 00 00 	movabs $0x1fffffffe0,%rax
   4eb44:	4c 21 c8             	and    %r9,%rax
   4eb47:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 4eb4e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24de>	4eb49: R_X86_64_PC32	.bss+0x1fb
   4eb4e:	0f 84 e2 00 00 00    	je     4ec36 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x25c6>
   4eb54:	48 c7 84 24 28 02 00 00 01 00 00 00 	movq   $0x1,0x228(%rsp)
   4eb60:	48 c7 84 24 30 02 00 00 04 00 00 00 	movq   $0x4,0x230(%rsp)
   4eb6c:	48 89 84 24 38 02 00 00 	mov    %rax,0x238(%rsp)
   4eb74:	48 c7 84 24 40 02 00 00 01 00 00 00 	movq   $0x1,0x240(%rsp)
   4eb80:	48 c7 84 24 48 02 00 00 04 00 00 00 	movq   $0x4,0x248(%rsp)
   4eb8c:	48 c7 84 24 50 02 00 00 20 00 00 00 	movq   $0x20,0x250(%rsp)
   4eb98:	0f 57 c0             	xorps  %xmm0,%xmm0
   4eb9b:	0f 11 84 24 58 02 00 00 	movups %xmm0,0x258(%rsp)
   4eba3:	48 c7 84 24 68 02 00 00 00 00 00 00 	movq   $0x0,0x268(%rsp)
   4ebaf:	48 89 bc 24 48 01 00 00 	mov    %rdi,0x148(%rsp)
   4ebb7:	48 89 b4 24 50 01 00 00 	mov    %rsi,0x150(%rsp)
   4ebbf:	48 89 94 24 58 01 00 00 	mov    %rdx,0x158(%rsp)
   4ebc7:	89 8c 24 60 01 00 00 	mov    %ecx,0x160(%rsp)
   4ebce:	44 89 84 24 64 01 00 00 	mov    %r8d,0x164(%rsp)
   4ebd6:	48 8b 84 24 68 02 00 00 	mov    0x268(%rsp),%rax
   4ebde:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   4ebe3:	0f 10 84 24 28 02 00 00 	movups 0x228(%rsp),%xmm0
   4ebeb:	0f 10 8c 24 38 02 00 00 	movups 0x238(%rsp),%xmm1
   4ebf3:	0f 10 94 24 48 02 00 00 	movups 0x248(%rsp),%xmm2
   4ebfb:	0f 10 9c 24 58 02 00 00 	movups 0x258(%rsp),%xmm3
   4ec03:	0f 11 5c 24 30       	movups %xmm3,0x30(%rsp)
   4ec08:	0f 11 54 24 20       	movups %xmm2,0x20(%rsp)
   4ec0d:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
   4ec12:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ec16:	4c 8d b4 24 68 04 00 00 	lea    0x468(%rsp),%r14
   4ec1e:	48 8d 94 24 48 01 00 00 	lea    0x148(%rsp),%rdx
   4ec26:	4c 89 f7             	mov    %r14,%rdi
   4ec29:	48 89 de             	mov    %rbx,%rsi
   4ec2c:	e8 9f 63 03 00       	call   84fd0 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_124native_mmvq_q6k_wide_a2rE18dpct_kernel_scalarILi6EES8_ILi1EEEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSE_PfiiPS1_EUlNS0_7nd_itemILi3EEEE9_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSL_>
   4ec31:	e9 dd 00 00 00       	jmp    4ed13 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26a3>
   4ec36:	48 c7 84 24 e0 01 00 00 01 00 00 00 	movq   $0x1,0x1e0(%rsp)
   4ec42:	48 c7 84 24 e8 01 00 00 04 00 00 00 	movq   $0x4,0x1e8(%rsp)
   4ec4e:	48 89 84 24 f0 01 00 00 	mov    %rax,0x1f0(%rsp)
   4ec56:	48 c7 84 24 f8 01 00 00 01 00 00 00 	movq   $0x1,0x1f8(%rsp)
   4ec62:	48 c7 84 24 00 02 00 00 04 00 00 00 	movq   $0x4,0x200(%rsp)
   4ec6e:	48 c7 84 24 08 02 00 00 20 00 00 00 	movq   $0x20,0x208(%rsp)
   4ec7a:	0f 57 c0             	xorps  %xmm0,%xmm0
   4ec7d:	0f 11 84 24 10 02 00 00 	movups %xmm0,0x210(%rsp)
   4ec85:	48 c7 84 24 20 02 00 00 00 00 00 00 	movq   $0x0,0x220(%rsp)
   4ec91:	48 89 bc 24 48 01 00 00 	mov    %rdi,0x148(%rsp)
   4ec99:	48 89 b4 24 50 01 00 00 	mov    %rsi,0x150(%rsp)
   4eca1:	48 89 94 24 58 01 00 00 	mov    %rdx,0x158(%rsp)
   4eca9:	89 8c 24 60 01 00 00 	mov    %ecx,0x160(%rsp)
   4ecb0:	44 89 84 24 64 01 00 00 	mov    %r8d,0x164(%rsp)
   4ecb8:	48 8b 84 24 20 02 00 00 	mov    0x220(%rsp),%rax
   4ecc0:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
   4ecc5:	0f 10 84 24 e0 01 00 00 	movups 0x1e0(%rsp),%xmm0
   4eccd:	0f 10 8c 24 f0 01 00 00 	movups 0x1f0(%rsp),%xmm1
   4ecd5:	0f 10 94 24 00 02 00 00 	movups 0x200(%rsp),%xmm2
   4ecdd:	0f 10 9c 24 10 02 00 00 	movups 0x210(%rsp),%xmm3
   4ece5:	0f 11 5c 24 30       	movups %xmm3,0x30(%rsp)
   4ecea:	0f 11 54 24 20       	movups %xmm2,0x20(%rsp)
   4ecef:	0f 11 4c 24 10       	movups %xmm1,0x10(%rsp)
   4ecf4:	0f 11 04 24          	movups %xmm0,(%rsp)
   4ecf8:	4c 8d b4 24 58 04 00 00 	lea    0x458(%rsp),%r14
   4ed00:	48 8d 94 24 48 01 00 00 	lea    0x148(%rsp),%rdx
   4ed08:	4c 89 f7             	mov    %r14,%rdi
   4ed0b:	48 89 de             	mov    %rbx,%rsi
   4ed0e:	e8 1d 65 03 00       	call   85230 <_ZN4sycl3_V15queue12parallel_forI16dpct_kernel_nameIJN6strata7kernels12_GLOBAL__N_120native_mmvq_q6k_wideE18dpct_kernel_scalarILi6EES8_ILi1EEEELi3EJZNS6_15launch_q6k_wideILi6EEEvPKvSE_PfiiPS1_EUlNS0_7nd_itemILi3EEEE10_EEENSt9enable_ifIXsr6detail23AreAllButLastReductionsIDpT1_EE5valueENS0_5eventEE4typeENS0_8nd_rangeIXT0_EEEDpOSL_>
   4ed13:	4c 89 f7             	mov    %r14,%rdi
   4ed16:	e8 00 00 00 00       	call   4ed1b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26ab>	4ed17: R_X86_64_PLT32	_ZN4sycl3_V15eventD2Ev-0x4
   4ed1b:	e9 f1 fd ff ff       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4ed20:	48 89 df             	mov    %rbx,%rdi
   4ed23:	e8 00 00 00 00       	call   4ed28 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26b8>	4ed24: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
   4ed28:	e9 e4 fd ff ff       	jmp    4eb11 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24a1>
   4ed2d:	49 89 fe             	mov    %rdi,%r14
   4ed30:	bf 00 00 00 00       	mov    $0x0,%edi	4ed31: R_X86_64_32	.bss+0x1948
   4ed35:	44 89 c5             	mov    %r8d,%ebp
   4ed38:	41 89 cf             	mov    %ecx,%r15d
   4ed3b:	49 89 d4             	mov    %rdx,%r12
   4ed3e:	49 89 f5             	mov    %rsi,%r13
   4ed41:	e8 00 00 00 00       	call   4ed46 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26d6>	4ed42: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ed46:	4c 89 f7             	mov    %r14,%rdi
   4ed49:	4c 89 ee             	mov    %r13,%rsi
   4ed4c:	4c 89 e2             	mov    %r12,%rdx
   4ed4f:	44 89 f9             	mov    %r15d,%ecx
   4ed52:	41 89 e8             	mov    %ebp,%r8d
   4ed55:	85 c0                	test   %eax,%eax
   4ed57:	0f 84 36 d9 ff ff    	je     4c693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23>
   4ed5d:	bf 00 00 00 00       	mov    $0x0,%edi	4ed5e: R_X86_64_32	.rodata.str1.1+0x1244d
   4ed62:	e8 00 00 00 00       	call   4ed67 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x26f7>	4ed63: R_X86_64_PLT32	getenv-0x4
   4ed67:	48 85 c0             	test   %rax,%rax
   4ed6a:	74 65                	je     4edd1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2761>
   4ed6c:	48 89 c7             	mov    %rax,%rdi
   4ed6f:	31 f6                	xor    %esi,%esi
   4ed71:	ba 0a 00 00 00       	mov    $0xa,%edx
   4ed76:	e8 00 00 00 00       	call   4ed7b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x270b>	4ed77: R_X86_64_PLT32	__isoc23_strtol-0x4
   4ed7b:	eb 59                	jmp    4edd6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2766>
   4ed7d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4ed82:	bf 00 00 00 00       	mov    $0x0,%edi	4ed83: R_X86_64_32	.bss+0x1958
   4ed87:	45 89 c7             	mov    %r8d,%r15d
   4ed8a:	41 89 cc             	mov    %ecx,%r12d
   4ed8d:	49 89 d5             	mov    %rdx,%r13
   4ed90:	49 89 f6             	mov    %rsi,%r14
   4ed93:	e8 00 00 00 00       	call   4ed98 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2728>	4ed94: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ed98:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ed9d:	4c 89 f6             	mov    %r14,%rsi
   4eda0:	4c 89 ea             	mov    %r13,%rdx
   4eda3:	44 89 e1             	mov    %r12d,%ecx
   4eda6:	45 89 f8             	mov    %r15d,%r8d
   4eda9:	85 c0                	test   %eax,%eax
   4edab:	0f 84 f7 d8 ff ff    	je     4c6a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x38>
   4edb1:	bf 00 00 00 00       	mov    $0x0,%edi	4edb2: R_X86_64_32	.rodata.str1.1+0x12462
   4edb6:	e8 00 00 00 00       	call   4edbb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x274b>	4edb7: R_X86_64_PLT32	getenv-0x4
   4edbb:	48 85 c0             	test   %rax,%rax
   4edbe:	74 3a                	je     4edfa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x278a>
   4edc0:	48 89 c7             	mov    %rax,%rdi
   4edc3:	31 f6                	xor    %esi,%esi
   4edc5:	ba 0a 00 00 00       	mov    $0xa,%edx
   4edca:	e8 00 00 00 00       	call   4edcf <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x275f>	4edcb: R_X86_64_PLT32	__isoc23_strtol-0x4
   4edcf:	eb 2b                	jmp    4edfc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x278c>
   4edd1:	b8 01 00 00 00       	mov    $0x1,%eax
   4edd6:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4eddc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x276c>	4edd8: R_X86_64_PC32	.bss+0x193c
   4eddc:	bf 00 00 00 00       	mov    $0x0,%edi	4eddd: R_X86_64_32	.bss+0x1948
   4ede1:	e8 00 00 00 00       	call   4ede6 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2776>	4ede2: R_X86_64_PLT32	__cxa_guard_release-0x4
   4ede6:	41 89 e8             	mov    %ebp,%r8d
   4ede9:	44 89 f9             	mov    %r15d,%ecx
   4edec:	4c 89 e2             	mov    %r12,%rdx
   4edef:	4c 89 ee             	mov    %r13,%rsi
   4edf2:	4c 89 f7             	mov    %r14,%rdi
   4edf5:	e9 99 d8 ff ff       	jmp    4c693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x23>
   4edfa:	31 c0                	xor    %eax,%eax
   4edfc:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4ee02 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2792>	4edfe: R_X86_64_PC32	.bss+0x194c
   4ee02:	bf 00 00 00 00       	mov    $0x0,%edi	4ee03: R_X86_64_32	.bss+0x1958
   4ee07:	e8 00 00 00 00       	call   4ee0c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x279c>	4ee08: R_X86_64_PLT32	__cxa_guard_release-0x4
   4ee0c:	45 89 f8             	mov    %r15d,%r8d
   4ee0f:	44 89 e1             	mov    %r12d,%ecx
   4ee12:	4c 89 ea             	mov    %r13,%rdx
   4ee15:	4c 89 f6             	mov    %r14,%rsi
   4ee18:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ee1d:	e9 86 d8 ff ff       	jmp    4c6a8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x38>
   4ee22:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4ee27:	bf 00 00 00 00       	mov    $0x0,%edi	4ee28: R_X86_64_32	.bss+0x208
   4ee2c:	44 89 c5             	mov    %r8d,%ebp
   4ee2f:	41 89 cc             	mov    %ecx,%r12d
   4ee32:	49 89 d5             	mov    %rdx,%r13
   4ee35:	49 89 f6             	mov    %rsi,%r14
   4ee38:	e8 00 00 00 00       	call   4ee3d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x27cd>	4ee39: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ee3d:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ee42:	4c 89 f6             	mov    %r14,%rsi
   4ee45:	4c 89 ea             	mov    %r13,%rdx
   4ee48:	44 89 e1             	mov    %r12d,%ecx
   4ee4b:	41 89 e8             	mov    %ebp,%r8d
   4ee4e:	85 c0                	test   %eax,%eax
   4ee50:	0f 84 84 dd ff ff    	je     4cbda <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x56a>
   4ee56:	bf 00 00 00 00       	mov    $0x0,%edi	4ee57: R_X86_64_32	.rodata.str1.1+0x1cb3
   4ee5b:	e8 00 00 00 00       	call   4ee60 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x27f0>	4ee5c: R_X86_64_PLT32	getenv-0x4
   4ee60:	48 85 c0             	test   %rax,%rax
   4ee63:	0f 84 9a 05 00 00    	je     4f403 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d93>
   4ee69:	48 89 c7             	mov    %rax,%rdi
   4ee6c:	31 f6                	xor    %esi,%esi
   4ee6e:	ba 0a 00 00 00       	mov    $0xa,%edx
   4ee73:	e8 00 00 00 00       	call   4ee78 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2808>	4ee74: R_X86_64_PLT32	__isoc23_strtol-0x4
   4ee78:	85 c0                	test   %eax,%eax
   4ee7a:	0f 95 c0             	setne  %al
   4ee7d:	e9 83 05 00 00       	jmp    4f405 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d95>
   4ee82:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4ee87:	bf 00 00 00 00       	mov    $0x0,%edi	4ee88: R_X86_64_32	.bss+0x208
   4ee8c:	44 89 c5             	mov    %r8d,%ebp
   4ee8f:	41 89 cc             	mov    %ecx,%r12d
   4ee92:	49 89 d5             	mov    %rdx,%r13
   4ee95:	49 89 f6             	mov    %rsi,%r14
   4ee98:	e8 00 00 00 00       	call   4ee9d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x282d>	4ee99: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ee9d:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4eea2:	4c 89 f6             	mov    %r14,%rsi
   4eea5:	4c 89 ea             	mov    %r13,%rdx
   4eea8:	44 89 e1             	mov    %r12d,%ecx
   4eeab:	41 89 e8             	mov    %ebp,%r8d
   4eeae:	85 c0                	test   %eax,%eax
   4eeb0:	0f 84 29 d8 ff ff    	je     4c6df <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6f>
   4eeb6:	bf 00 00 00 00       	mov    $0x0,%edi	4eeb7: R_X86_64_32	.rodata.str1.1+0x1cb3
   4eebb:	e8 00 00 00 00       	call   4eec0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2850>	4eebc: R_X86_64_PLT32	getenv-0x4
   4eec0:	48 85 c0             	test   %rax,%rax
   4eec3:	0f 84 62 05 00 00    	je     4f42b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dbb>
   4eec9:	48 89 c7             	mov    %rax,%rdi
   4eecc:	31 f6                	xor    %esi,%esi
   4eece:	ba 0a 00 00 00       	mov    $0xa,%edx
   4eed3:	e8 00 00 00 00       	call   4eed8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2868>	4eed4: R_X86_64_PLT32	__isoc23_strtol-0x4
   4eed8:	85 c0                	test   %eax,%eax
   4eeda:	0f 95 c0             	setne  %al
   4eedd:	e9 4b 05 00 00       	jmp    4f42d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dbd>
   4eee2:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4eee7:	bf 00 00 00 00       	mov    $0x0,%edi	4eee8: R_X86_64_32	.bss+0x208
   4eeec:	44 89 c5             	mov    %r8d,%ebp
   4eeef:	41 89 cc             	mov    %ecx,%r12d
   4eef2:	49 89 d5             	mov    %rdx,%r13
   4eef5:	49 89 f6             	mov    %rsi,%r14
   4eef8:	e8 00 00 00 00       	call   4eefd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x288d>	4eef9: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4eefd:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ef02:	4c 89 f6             	mov    %r14,%rsi
   4ef05:	4c 89 ea             	mov    %r13,%rdx
   4ef08:	44 89 e1             	mov    %r12d,%ecx
   4ef0b:	41 89 e8             	mov    %ebp,%r8d
   4ef0e:	85 c0                	test   %eax,%eax
   4ef10:	0f 84 3f df ff ff    	je     4ce55 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7e5>
   4ef16:	bf 00 00 00 00       	mov    $0x0,%edi	4ef17: R_X86_64_32	.rodata.str1.1+0x1cb3
   4ef1b:	e8 00 00 00 00       	call   4ef20 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28b0>	4ef1c: R_X86_64_PLT32	getenv-0x4
   4ef20:	48 85 c0             	test   %rax,%rax
   4ef23:	0f 84 2a 05 00 00    	je     4f453 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2de3>
   4ef29:	48 89 c7             	mov    %rax,%rdi
   4ef2c:	31 f6                	xor    %esi,%esi
   4ef2e:	ba 0a 00 00 00       	mov    $0xa,%edx
   4ef33:	e8 00 00 00 00       	call   4ef38 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28c8>	4ef34: R_X86_64_PLT32	__isoc23_strtol-0x4
   4ef38:	85 c0                	test   %eax,%eax
   4ef3a:	0f 95 c0             	setne  %al
   4ef3d:	e9 13 05 00 00       	jmp    4f455 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2de5>
   4ef42:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4ef47:	bf 00 00 00 00       	mov    $0x0,%edi	4ef48: R_X86_64_32	.bss+0x208
   4ef4c:	44 89 c5             	mov    %r8d,%ebp
   4ef4f:	41 89 cc             	mov    %ecx,%r12d
   4ef52:	49 89 d5             	mov    %rdx,%r13
   4ef55:	49 89 f6             	mov    %rsi,%r14
   4ef58:	e8 00 00 00 00       	call   4ef5d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x28ed>	4ef59: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4ef5d:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4ef62:	4c 89 f6             	mov    %r14,%rsi
   4ef65:	4c 89 ea             	mov    %r13,%rdx
   4ef68:	44 89 e1             	mov    %r12d,%ecx
   4ef6b:	41 89 e8             	mov    %ebp,%r8d
   4ef6e:	85 c0                	test   %eax,%eax
   4ef70:	0f 84 5a e1 ff ff    	je     4d0d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa60>
   4ef76:	bf 00 00 00 00       	mov    $0x0,%edi	4ef77: R_X86_64_32	.rodata.str1.1+0x1cb3
   4ef7b:	e8 00 00 00 00       	call   4ef80 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2910>	4ef7c: R_X86_64_PLT32	getenv-0x4
   4ef80:	48 85 c0             	test   %rax,%rax
   4ef83:	0f 84 f2 04 00 00    	je     4f47b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e0b>
   4ef89:	48 89 c7             	mov    %rax,%rdi
   4ef8c:	31 f6                	xor    %esi,%esi
   4ef8e:	ba 0a 00 00 00       	mov    $0xa,%edx
   4ef93:	e8 00 00 00 00       	call   4ef98 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2928>	4ef94: R_X86_64_PLT32	__isoc23_strtol-0x4
   4ef98:	85 c0                	test   %eax,%eax
   4ef9a:	0f 95 c0             	setne  %al
   4ef9d:	e9 db 04 00 00       	jmp    4f47d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e0d>
   4efa2:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4efa7:	bf 00 00 00 00       	mov    $0x0,%edi	4efa8: R_X86_64_32	.bss+0x208
   4efac:	44 89 c5             	mov    %r8d,%ebp
   4efaf:	41 89 cc             	mov    %ecx,%r12d
   4efb2:	49 89 d5             	mov    %rdx,%r13
   4efb5:	49 89 f6             	mov    %rsi,%r14
   4efb8:	e8 00 00 00 00       	call   4efbd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x294d>	4efb9: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4efbd:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4efc2:	4c 89 f6             	mov    %r14,%rsi
   4efc5:	4c 89 ea             	mov    %r13,%rdx
   4efc8:	44 89 e1             	mov    %r12d,%ecx
   4efcb:	41 89 e8             	mov    %ebp,%r8d
   4efce:	85 c0                	test   %eax,%eax
   4efd0:	0f 84 78 e3 ff ff    	je     4d34e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcde>
   4efd6:	bf 00 00 00 00       	mov    $0x0,%edi	4efd7: R_X86_64_32	.rodata.str1.1+0x1cb3
   4efdb:	e8 00 00 00 00       	call   4efe0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2970>	4efdc: R_X86_64_PLT32	getenv-0x4
   4efe0:	48 85 c0             	test   %rax,%rax
   4efe3:	0f 84 ba 04 00 00    	je     4f4a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e33>
   4efe9:	48 89 c7             	mov    %rax,%rdi
   4efec:	31 f6                	xor    %esi,%esi
   4efee:	ba 0a 00 00 00       	mov    $0xa,%edx
   4eff3:	e8 00 00 00 00       	call   4eff8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2988>	4eff4: R_X86_64_PLT32	__isoc23_strtol-0x4
   4eff8:	85 c0                	test   %eax,%eax
   4effa:	0f 95 c0             	setne  %al
   4effd:	e9 a3 04 00 00       	jmp    4f4a5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e35>
   4f002:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f007:	bf 00 00 00 00       	mov    $0x0,%edi	4f008: R_X86_64_32	.bss+0x208
   4f00c:	44 89 c5             	mov    %r8d,%ebp
   4f00f:	41 89 cc             	mov    %ecx,%r12d
   4f012:	49 89 d5             	mov    %rdx,%r13
   4f015:	49 89 f6             	mov    %rsi,%r14
   4f018:	e8 00 00 00 00       	call   4f01d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29ad>	4f019: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f01d:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f022:	4c 89 f6             	mov    %r14,%rsi
   4f025:	4c 89 ea             	mov    %r13,%rdx
   4f028:	44 89 e1             	mov    %r12d,%ecx
   4f02b:	41 89 e8             	mov    %ebp,%r8d
   4f02e:	85 c0                	test   %eax,%eax
   4f030:	0f 84 7a e5 ff ff    	je     4d5b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf40>
   4f036:	bf 00 00 00 00       	mov    $0x0,%edi	4f037: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f03b:	e8 00 00 00 00       	call   4f040 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29d0>	4f03c: R_X86_64_PLT32	getenv-0x4
   4f040:	48 85 c0             	test   %rax,%rax
   4f043:	0f 84 82 04 00 00    	je     4f4cb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e5b>
   4f049:	48 89 c7             	mov    %rax,%rdi
   4f04c:	31 f6                	xor    %esi,%esi
   4f04e:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f053:	e8 00 00 00 00       	call   4f058 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x29e8>	4f054: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f058:	85 c0                	test   %eax,%eax
   4f05a:	0f 95 c0             	setne  %al
   4f05d:	e9 6b 04 00 00       	jmp    4f4cd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e5d>
   4f062:	49 89 fe             	mov    %rdi,%r14
   4f065:	bf 00 00 00 00       	mov    $0x0,%edi	4f066: R_X86_64_32	.bss+0x208
   4f06a:	44 89 c5             	mov    %r8d,%ebp
   4f06d:	41 89 cf             	mov    %ecx,%r15d
   4f070:	49 89 d4             	mov    %rdx,%r12
   4f073:	49 89 f5             	mov    %rsi,%r13
   4f076:	e8 00 00 00 00       	call   4f07b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a0b>	4f077: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f07b:	4c 89 f7             	mov    %r14,%rdi
   4f07e:	4c 89 ee             	mov    %r13,%rsi
   4f081:	4c 89 e2             	mov    %r12,%rdx
   4f084:	44 89 f9             	mov    %r15d,%ecx
   4f087:	41 89 e8             	mov    %ebp,%r8d
   4f08a:	85 c0                	test   %eax,%eax
   4f08c:	0f 84 e3 d8 ff ff    	je     4c975 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x305>
   4f092:	bf 00 00 00 00       	mov    $0x0,%edi	4f093: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f097:	e8 00 00 00 00       	call   4f09c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a2c>	4f098: R_X86_64_PLT32	getenv-0x4
   4f09c:	48 85 c0             	test   %rax,%rax
   4f09f:	0f 84 4e 04 00 00    	je     4f4f3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e83>
   4f0a5:	48 89 c7             	mov    %rax,%rdi
   4f0a8:	31 f6                	xor    %esi,%esi
   4f0aa:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f0af:	e8 00 00 00 00       	call   4f0b4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a44>	4f0b0: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f0b4:	85 c0                	test   %eax,%eax
   4f0b6:	0f 95 c0             	setne  %al
   4f0b9:	e9 37 04 00 00       	jmp    4f4f5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e85>
   4f0be:	bf 00 00 00 00       	mov    $0x0,%edi	4f0bf: R_X86_64_32	.bss+0x1fb8
   4f0c3:	e8 00 00 00 00       	call   4f0c8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a58>	4f0c4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f0c8:	85 c0                	test   %eax,%eax
   4f0ca:	0f 84 64 e8 ff ff    	je     4d934 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12c4>
   4f0d0:	bf 00 00 00 00       	mov    $0x0,%edi	4f0d1: R_X86_64_32	.rodata+0x491e0
   4f0d5:	e8 00 00 00 00       	call   4f0da <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a6a>	4f0d6: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f0da:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f0e1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a71>	4f0dd: R_X86_64_PC32	.bss+0x1fac
   4f0e1:	bf 00 00 00 00       	mov    $0x0,%edi	4f0e2: R_X86_64_32	.bss+0x1fb8
   4f0e6:	e8 00 00 00 00       	call   4f0eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a7b>	4f0e7: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f0eb:	e9 44 e8 ff ff       	jmp    4d934 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x12c4>
   4f0f0:	bf 00 00 00 00       	mov    $0x0,%edi	4f0f1: R_X86_64_32	.bss+0x1ef8
   4f0f5:	e8 00 00 00 00       	call   4f0fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a8a>	4f0f6: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f0fa:	85 c0                	test   %eax,%eax
   4f0fc:	0f 84 30 dc ff ff    	je     4cd32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6c2>
   4f102:	bf 00 00 00 00       	mov    $0x0,%edi	4f103: R_X86_64_32	.rodata+0x47980
   4f107:	e8 00 00 00 00       	call   4f10c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2a9c>	4f108: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f10c:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f113 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2aa3>	4f10f: R_X86_64_PC32	.bss+0x1eec
   4f113:	bf 00 00 00 00       	mov    $0x0,%edi	4f114: R_X86_64_32	.bss+0x1ef8
   4f118:	e8 00 00 00 00       	call   4f11d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2aad>	4f119: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f11d:	e9 10 dc ff ff       	jmp    4cd32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6c2>
   4f122:	bf 00 00 00 00       	mov    $0x0,%edi	4f123: R_X86_64_32	.bss+0x1eb8
   4f127:	e8 00 00 00 00       	call   4f12c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2abc>	4f128: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f12c:	85 c0                	test   %eax,%eax
   4f12e:	0f 84 03 d7 ff ff    	je     4c837 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c7>
   4f134:	bf 00 00 00 00       	mov    $0x0,%edi	4f135: R_X86_64_32	.rodata+0x47160
   4f139:	e8 00 00 00 00       	call   4f13e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ace>	4f13a: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f13e:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f145 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ad5>	4f141: R_X86_64_PC32	.bss+0x1eac
   4f145:	bf 00 00 00 00       	mov    $0x0,%edi	4f146: R_X86_64_32	.bss+0x1eb8
   4f14a:	e8 00 00 00 00       	call   4f14f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2adf>	4f14b: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f14f:	e9 e3 d6 ff ff       	jmp    4c837 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c7>
   4f154:	bf 00 00 00 00       	mov    $0x0,%edi	4f155: R_X86_64_32	.bss+0x1ed8
   4f159:	e8 00 00 00 00       	call   4f15e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2aee>	4f15a: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f15e:	85 c0                	test   %eax,%eax
   4f160:	0f 84 47 de ff ff    	je     4cfad <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x93d>
   4f166:	bf 00 00 00 00       	mov    $0x0,%edi	4f167: R_X86_64_32	.rodata+0x47570
   4f16b:	e8 00 00 00 00       	call   4f170 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b00>	4f16c: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f170:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f177 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b07>	4f173: R_X86_64_PC32	.bss+0x1ecc
   4f177:	bf 00 00 00 00       	mov    $0x0,%edi	4f178: R_X86_64_32	.bss+0x1ed8
   4f17c:	e8 00 00 00 00       	call   4f181 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b11>	4f17d: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f181:	e9 27 de ff ff       	jmp    4cfad <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x93d>
   4f186:	bf 00 00 00 00       	mov    $0x0,%edi	4f187: R_X86_64_32	.bss+0x1f18
   4f18b:	e8 00 00 00 00       	call   4f190 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b20>	4f18c: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f190:	85 c0                	test   %eax,%eax
   4f192:	0f 84 90 e0 ff ff    	je     4d228 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbb8>
   4f198:	bf 00 00 00 00       	mov    $0x0,%edi	4f199: R_X86_64_32	.rodata+0x47d90
   4f19d:	e8 00 00 00 00       	call   4f1a2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b32>	4f19e: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f1a2:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f1a9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b39>	4f1a5: R_X86_64_PC32	.bss+0x1f0c
   4f1a9:	bf 00 00 00 00       	mov    $0x0,%edi	4f1aa: R_X86_64_32	.bss+0x1f18
   4f1ae:	e8 00 00 00 00       	call   4f1b3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b43>	4f1af: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f1b3:	e9 70 e0 ff ff       	jmp    4d228 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xbb8>
   4f1b8:	bf 00 00 00 00       	mov    $0x0,%edi	4f1b9: R_X86_64_32	.bss+0x1f78
   4f1bd:	e8 00 00 00 00       	call   4f1c2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b52>	4f1be: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f1c2:	85 c0                	test   %eax,%eax
   4f1c4:	0f 84 c0 e2 ff ff    	je     4d48a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe1a>
   4f1ca:	bf 00 00 00 00       	mov    $0x0,%edi	4f1cb: R_X86_64_32	.rodata+0x489c0
   4f1cf:	e8 00 00 00 00       	call   4f1d4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b64>	4f1d0: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f1d4:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f1db <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b6b>	4f1d7: R_X86_64_PC32	.bss+0x1f6c
   4f1db:	bf 00 00 00 00       	mov    $0x0,%edi	4f1dc: R_X86_64_32	.bss+0x1f78
   4f1e0:	e8 00 00 00 00       	call   4f1e5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b75>	4f1e1: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f1e5:	e9 a0 e2 ff ff       	jmp    4d48a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xe1a>
   4f1ea:	bf 00 00 00 00       	mov    $0x0,%edi	4f1eb: R_X86_64_32	.bss+0x1f38
   4f1ef:	e8 00 00 00 00       	call   4f1f4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b84>	4f1f0: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f1f4:	85 c0                	test   %eax,%eax
   4f1f6:	0f 84 bb d8 ff ff    	je     4cab7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x447>
   4f1fc:	bf 00 00 00 00       	mov    $0x0,%edi	4f1fd: R_X86_64_32	.rodata+0x481a0
   4f201:	e8 00 00 00 00       	call   4f206 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b96>	4f202: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f206:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f20d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2b9d>	4f209: R_X86_64_PC32	.bss+0x1f2c
   4f20d:	bf 00 00 00 00       	mov    $0x0,%edi	4f20e: R_X86_64_32	.bss+0x1f38
   4f212:	e8 00 00 00 00       	call   4f217 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ba7>	4f213: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f217:	e9 9b d8 ff ff       	jmp    4cab7 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x447>
   4f21c:	bf 00 00 00 00       	mov    $0x0,%edi	4f21d: R_X86_64_32	.bss+0x1f98
   4f221:	e8 00 00 00 00       	call   4f226 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bb6>	4f222: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f226:	85 c0                	test   %eax,%eax
   4f228:	0f 84 bd e4 ff ff    	je     4d6eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x107b>
   4f22e:	bf 00 00 00 00       	mov    $0x0,%edi	4f22f: R_X86_64_32	.rodata+0x48dd0
   4f233:	e8 00 00 00 00       	call   4f238 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bc8>	4f234: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f238:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f23f <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bcf>	4f23b: R_X86_64_PC32	.bss+0x1f8c
   4f23f:	bf 00 00 00 00       	mov    $0x0,%edi	4f240: R_X86_64_32	.bss+0x1f98
   4f244:	e8 00 00 00 00       	call   4f249 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bd9>	4f245: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f249:	e9 9d e4 ff ff       	jmp    4d6eb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x107b>
   4f24e:	49 89 fe             	mov    %rdi,%r14
   4f251:	bf 00 00 00 00       	mov    $0x0,%edi	4f252: R_X86_64_32	.bss+0x4d8
   4f256:	44 89 c5             	mov    %r8d,%ebp
   4f259:	41 89 cf             	mov    %ecx,%r15d
   4f25c:	49 89 d4             	mov    %rdx,%r12
   4f25f:	49 89 f5             	mov    %rsi,%r13
   4f262:	e8 00 00 00 00       	call   4f267 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2bf7>	4f263: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f267:	4c 89 f7             	mov    %r14,%rdi
   4f26a:	4c 89 ee             	mov    %r13,%rsi
   4f26d:	4c 89 e2             	mov    %r12,%rdx
   4f270:	44 89 f9             	mov    %r15d,%ecx
   4f273:	41 89 e8             	mov    %ebp,%r8d
   4f276:	85 c0                	test   %eax,%eax
   4f278:	0f 84 d2 f5 ff ff    	je     4e850 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21e0>
   4f27e:	bf 00 00 00 00       	mov    $0x0,%edi	4f27f: R_X86_64_32	.rodata.str1.1+0x3bcf
   4f283:	e8 00 00 00 00       	call   4f288 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c18>	4f284: R_X86_64_PLT32	getenv-0x4
   4f288:	48 85 c0             	test   %rax,%rax
   4f28b:	0f 84 e1 02 00 00    	je     4f572 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f02>
   4f291:	48 89 c7             	mov    %rax,%rdi
   4f294:	31 f6                	xor    %esi,%esi
   4f296:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f29b:	e8 00 00 00 00       	call   4f2a0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c30>	4f29c: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f2a0:	e9 d2 02 00 00       	jmp    4f577 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f07>
   4f2a5:	bf 00 00 00 00       	mov    $0x0,%edi	4f2a6: R_X86_64_32	.bss+0x1f08
   4f2aa:	e8 00 00 00 00       	call   4f2af <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c3f>	4f2ab: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f2af:	85 c0                	test   %eax,%eax
   4f2b1:	0f 84 dc e8 ff ff    	je     4db93 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1523>
   4f2b7:	bf 00 00 00 00       	mov    $0x0,%edi	4f2b8: R_X86_64_32	.rodata+0x47b88
   4f2bc:	e8 00 00 00 00       	call   4f2c1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c51>	4f2bd: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f2c1:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f2c8 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c58>	4f2c4: R_X86_64_PC32	.bss+0x1efc
   4f2c8:	bf 00 00 00 00       	mov    $0x0,%edi	4f2c9: R_X86_64_32	.bss+0x1f08
   4f2cd:	e8 00 00 00 00       	call   4f2d2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c62>	4f2ce: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f2d2:	e9 bc e8 ff ff       	jmp    4db93 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1523>
   4f2d7:	bf 00 00 00 00       	mov    $0x0,%edi	4f2d8: R_X86_64_32	.bss+0x1ec8
   4f2dc:	e8 00 00 00 00       	call   4f2e1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c71>	4f2dd: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f2e1:	85 c0                	test   %eax,%eax
   4f2e3:	0f 84 09 eb ff ff    	je     4ddf2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1782>
   4f2e9:	bf 00 00 00 00       	mov    $0x0,%edi	4f2ea: R_X86_64_32	.rodata+0x47368
   4f2ee:	e8 00 00 00 00       	call   4f2f3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c83>	4f2ef: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f2f3:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f2fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c8a>	4f2f6: R_X86_64_PC32	.bss+0x1ebc
   4f2fa:	bf 00 00 00 00       	mov    $0x0,%edi	4f2fb: R_X86_64_32	.bss+0x1ec8
   4f2ff:	e8 00 00 00 00       	call   4f304 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2c94>	4f300: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f304:	e9 e9 ea ff ff       	jmp    4ddf2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1782>
   4f309:	bf 00 00 00 00       	mov    $0x0,%edi	4f30a: R_X86_64_32	.bss+0x1ee8
   4f30e:	e8 00 00 00 00       	call   4f313 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ca3>	4f30f: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f313:	85 c0                	test   %eax,%eax
   4f315:	0f 84 36 ed ff ff    	je     4e051 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19e1>
   4f31b:	bf 00 00 00 00       	mov    $0x0,%edi	4f31c: R_X86_64_32	.rodata+0x47778
   4f320:	e8 00 00 00 00       	call   4f325 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cb5>	4f321: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f325:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f32c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cbc>	4f328: R_X86_64_PC32	.bss+0x1edc
   4f32c:	bf 00 00 00 00       	mov    $0x0,%edi	4f32d: R_X86_64_32	.bss+0x1ee8
   4f331:	e8 00 00 00 00       	call   4f336 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cc6>	4f332: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f336:	e9 16 ed ff ff       	jmp    4e051 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x19e1>
   4f33b:	bf 00 00 00 00       	mov    $0x0,%edi	4f33c: R_X86_64_32	.bss+0x1f28
   4f340:	e8 00 00 00 00       	call   4f345 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cd5>	4f341: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f345:	85 c0                	test   %eax,%eax
   4f347:	0f 84 63 ef ff ff    	je     4e2b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c40>
   4f34d:	bf 00 00 00 00       	mov    $0x0,%edi	4f34e: R_X86_64_32	.rodata+0x47f98
   4f352:	e8 00 00 00 00       	call   4f357 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ce7>	4f353: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f357:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f35e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cee>	4f35a: R_X86_64_PC32	.bss+0x1f1c
   4f35e:	bf 00 00 00 00       	mov    $0x0,%edi	4f35f: R_X86_64_32	.bss+0x1f28
   4f363:	e8 00 00 00 00       	call   4f368 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2cf8>	4f364: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f368:	e9 43 ef ff ff       	jmp    4e2b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1c40>
   4f36d:	bf 00 00 00 00       	mov    $0x0,%edi	4f36e: R_X86_64_32	.bss+0x1f88
   4f372:	e8 00 00 00 00       	call   4f377 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d07>	4f373: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f377:	85 c0                	test   %eax,%eax
   4f379:	0f 84 70 f1 ff ff    	je     4e4ef <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e7f>
   4f37f:	bf 00 00 00 00       	mov    $0x0,%edi	4f380: R_X86_64_32	.rodata+0x48bc8
   4f384:	e8 00 00 00 00       	call   4f389 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d19>	4f385: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f389:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f390 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d20>	4f38c: R_X86_64_PC32	.bss+0x1f7c
   4f390:	bf 00 00 00 00       	mov    $0x0,%edi	4f391: R_X86_64_32	.bss+0x1f88
   4f395:	e8 00 00 00 00       	call   4f39a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d2a>	4f396: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f39a:	e9 50 f1 ff ff       	jmp    4e4ef <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x1e7f>
   4f39f:	bf 00 00 00 00       	mov    $0x0,%edi	4f3a0: R_X86_64_32	.bss+0x1fa8
   4f3a4:	e8 00 00 00 00       	call   4f3a9 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d39>	4f3a5: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f3a9:	85 c0                	test   %eax,%eax
   4f3ab:	0f 84 7c f3 ff ff    	je     4e72d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20bd>
   4f3b1:	bf 00 00 00 00       	mov    $0x0,%edi	4f3b2: R_X86_64_32	.rodata+0x48fd8
   4f3b6:	e8 00 00 00 00       	call   4f3bb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d4b>	4f3b7: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f3bb:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f3c2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d52>	4f3be: R_X86_64_PC32	.bss+0x1f9c
   4f3c2:	bf 00 00 00 00       	mov    $0x0,%edi	4f3c3: R_X86_64_32	.bss+0x1fa8
   4f3c7:	e8 00 00 00 00       	call   4f3cc <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d5c>	4f3c8: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f3cc:	e9 5c f3 ff ff       	jmp    4e72d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x20bd>
   4f3d1:	bf 00 00 00 00       	mov    $0x0,%edi	4f3d2: R_X86_64_32	.bss+0x1f48
   4f3d6:	e8 00 00 00 00       	call   4f3db <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d6b>	4f3d7: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f3db:	85 c0                	test   %eax,%eax
   4f3dd:	0f 84 c8 f5 ff ff    	je     4e9ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x233b>
   4f3e3:	bf 00 00 00 00       	mov    $0x0,%edi	4f3e4: R_X86_64_32	.rodata+0x483a8
   4f3e8:	e8 00 00 00 00       	call   4f3ed <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d7d>	4f3e9: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   4f3ed:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 4f3f4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d84>	4f3f0: R_X86_64_PC32	.bss+0x1f3c
   4f3f4:	bf 00 00 00 00       	mov    $0x0,%edi	4f3f5: R_X86_64_32	.bss+0x1f48
   4f3f9:	e8 00 00 00 00       	call   4f3fe <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d8e>	4f3fa: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f3fe:	e9 a8 f5 ff ff       	jmp    4e9ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x233b>
   4f403:	b0 01                	mov    $0x1,%al
   4f405:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f40b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2d9b>	4f407: R_X86_64_PC32	.bss+0x1fc
   4f40b:	bf 00 00 00 00       	mov    $0x0,%edi	4f40c: R_X86_64_32	.bss+0x208
   4f410:	e8 00 00 00 00       	call   4f415 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2da5>	4f411: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f415:	41 89 e8             	mov    %ebp,%r8d
   4f418:	44 89 e1             	mov    %r12d,%ecx
   4f41b:	4c 89 ea             	mov    %r13,%rdx
   4f41e:	4c 89 f6             	mov    %r14,%rsi
   4f421:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f426:	e9 af d7 ff ff       	jmp    4cbda <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x56a>
   4f42b:	b0 01                	mov    $0x1,%al
   4f42d:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f433 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dc3>	4f42f: R_X86_64_PC32	.bss+0x1fc
   4f433:	bf 00 00 00 00       	mov    $0x0,%edi	4f434: R_X86_64_32	.bss+0x208
   4f438:	e8 00 00 00 00       	call   4f43d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2dcd>	4f439: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f43d:	41 89 e8             	mov    %ebp,%r8d
   4f440:	44 89 e1             	mov    %r12d,%ecx
   4f443:	4c 89 ea             	mov    %r13,%rdx
   4f446:	4c 89 f6             	mov    %r14,%rsi
   4f449:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f44e:	e9 8c d2 ff ff       	jmp    4c6df <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x6f>
   4f453:	b0 01                	mov    $0x1,%al
   4f455:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f45b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2deb>	4f457: R_X86_64_PC32	.bss+0x1fc
   4f45b:	bf 00 00 00 00       	mov    $0x0,%edi	4f45c: R_X86_64_32	.bss+0x208
   4f460:	e8 00 00 00 00       	call   4f465 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2df5>	4f461: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f465:	41 89 e8             	mov    %ebp,%r8d
   4f468:	44 89 e1             	mov    %r12d,%ecx
   4f46b:	4c 89 ea             	mov    %r13,%rdx
   4f46e:	4c 89 f6             	mov    %r14,%rsi
   4f471:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f476:	e9 da d9 ff ff       	jmp    4ce55 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x7e5>
   4f47b:	b0 01                	mov    $0x1,%al
   4f47d:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f483 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e13>	4f47f: R_X86_64_PC32	.bss+0x1fc
   4f483:	bf 00 00 00 00       	mov    $0x0,%edi	4f484: R_X86_64_32	.bss+0x208
   4f488:	e8 00 00 00 00       	call   4f48d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e1d>	4f489: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f48d:	41 89 e8             	mov    %ebp,%r8d
   4f490:	44 89 e1             	mov    %r12d,%ecx
   4f493:	4c 89 ea             	mov    %r13,%rdx
   4f496:	4c 89 f6             	mov    %r14,%rsi
   4f499:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f49e:	e9 2d dc ff ff       	jmp    4d0d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xa60>
   4f4a3:	b0 01                	mov    $0x1,%al
   4f4a5:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f4ab <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e3b>	4f4a7: R_X86_64_PC32	.bss+0x1fc
   4f4ab:	bf 00 00 00 00       	mov    $0x0,%edi	4f4ac: R_X86_64_32	.bss+0x208
   4f4b0:	e8 00 00 00 00       	call   4f4b5 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e45>	4f4b1: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f4b5:	41 89 e8             	mov    %ebp,%r8d
   4f4b8:	44 89 e1             	mov    %r12d,%ecx
   4f4bb:	4c 89 ea             	mov    %r13,%rdx
   4f4be:	4c 89 f6             	mov    %r14,%rsi
   4f4c1:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f4c6:	e9 83 de ff ff       	jmp    4d34e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xcde>
   4f4cb:	b0 01                	mov    $0x1,%al
   4f4cd:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f4d3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e63>	4f4cf: R_X86_64_PC32	.bss+0x1fc
   4f4d3:	bf 00 00 00 00       	mov    $0x0,%edi	4f4d4: R_X86_64_32	.bss+0x208
   4f4d8:	e8 00 00 00 00       	call   4f4dd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e6d>	4f4d9: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f4dd:	41 89 e8             	mov    %ebp,%r8d
   4f4e0:	44 89 e1             	mov    %r12d,%ecx
   4f4e3:	4c 89 ea             	mov    %r13,%rdx
   4f4e6:	4c 89 f6             	mov    %r14,%rsi
   4f4e9:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f4ee:	e9 bd e0 ff ff       	jmp    4d5b0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0xf40>
   4f4f3:	b0 01                	mov    $0x1,%al
   4f4f5:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f4fb <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e8b>	4f4f7: R_X86_64_PC32	.bss+0x1fc
   4f4fb:	bf 00 00 00 00       	mov    $0x0,%edi	4f4fc: R_X86_64_32	.bss+0x208
   4f500:	e8 00 00 00 00       	call   4f505 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2e95>	4f501: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f505:	41 89 e8             	mov    %ebp,%r8d
   4f508:	44 89 f9             	mov    %r15d,%ecx
   4f50b:	4c 89 e2             	mov    %r12,%rdx
   4f50e:	4c 89 ee             	mov    %r13,%rsi
   4f511:	4c 89 f7             	mov    %r14,%rdi
   4f514:	e9 5c d4 ff ff       	jmp    4c975 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x305>
   4f519:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
   4f51e:	bf 00 00 00 00       	mov    $0x0,%edi	4f51f: R_X86_64_32	.bss+0x208
   4f523:	44 89 c5             	mov    %r8d,%ebp
   4f526:	41 89 cc             	mov    %ecx,%r12d
   4f529:	49 89 d5             	mov    %rdx,%r13
   4f52c:	49 89 f6             	mov    %rsi,%r14
   4f52f:	e8 00 00 00 00       	call   4f534 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ec4>	4f530: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   4f534:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f539:	4c 89 f6             	mov    %r14,%rsi
   4f53c:	4c 89 ea             	mov    %r13,%rdx
   4f53f:	44 89 e1             	mov    %r12d,%ecx
   4f542:	41 89 e8             	mov    %ebp,%r8d
   4f545:	85 c0                	test   %eax,%eax
   4f547:	0f 84 e5 f5 ff ff    	je     4eb32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24c2>
   4f54d:	bf 00 00 00 00       	mov    $0x0,%edi	4f54e: R_X86_64_32	.rodata.str1.1+0x1cb3
   4f552:	e8 00 00 00 00       	call   4f557 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ee7>	4f553: R_X86_64_PLT32	getenv-0x4
   4f557:	48 85 c0             	test   %rax,%rax
   4f55a:	74 3f                	je     4f59b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f2b>
   4f55c:	48 89 c7             	mov    %rax,%rdi
   4f55f:	31 f6                	xor    %esi,%esi
   4f561:	ba 0a 00 00 00       	mov    $0xa,%edx
   4f566:	e8 00 00 00 00       	call   4f56b <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2efb>	4f567: R_X86_64_PLT32	__isoc23_strtol-0x4
   4f56b:	85 c0                	test   %eax,%eax
   4f56d:	0f 95 c0             	setne  %al
   4f570:	eb 2b                	jmp    4f59d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f2d>
   4f572:	b8 20 00 00 00       	mov    $0x20,%eax
   4f577:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 4f57d <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f0d>	4f579: R_X86_64_PC32	.bss+0x4cc
   4f57d:	bf 00 00 00 00       	mov    $0x0,%edi	4f57e: R_X86_64_32	.bss+0x4d8
   4f582:	e8 00 00 00 00       	call   4f587 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f17>	4f583: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f587:	41 89 e8             	mov    %ebp,%r8d
   4f58a:	44 89 f9             	mov    %r15d,%ecx
   4f58d:	4c 89 e2             	mov    %r12,%rdx
   4f590:	4c 89 ee             	mov    %r13,%rsi
   4f593:	4c 89 f7             	mov    %r14,%rdi
   4f596:	e9 b5 f2 ff ff       	jmp    4e850 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x21e0>
   4f59b:	b0 01                	mov    $0x1,%al
   4f59d:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 4f5a3 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f33>	4f59f: R_X86_64_PC32	.bss+0x1fc
   4f5a3:	bf 00 00 00 00       	mov    $0x0,%edi	4f5a4: R_X86_64_32	.bss+0x208
   4f5a8:	e8 00 00 00 00       	call   4f5ad <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f3d>	4f5a9: R_X86_64_PLT32	__cxa_guard_release-0x4
   4f5ad:	41 89 e8             	mov    %ebp,%r8d
   4f5b0:	44 89 e1             	mov    %r12d,%ecx
   4f5b3:	4c 89 ea             	mov    %r13,%rdx
   4f5b6:	4c 89 f6             	mov    %r14,%rsi
   4f5b9:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
   4f5be:	e9 6f f5 ff ff       	jmp    4eb32 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x24c2>
   4f5c3:	48 89 c3             	mov    %rax,%rbx
   4f5c6:	bf 00 00 00 00       	mov    $0x0,%edi	4f5c7: R_X86_64_32	.bss+0x1f48
   4f5cb:	e8 00 00 00 00       	call   4f5d0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f60>	4f5cc: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f5d0:	e9 28 01 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f5d5:	48 89 c3             	mov    %rax,%rbx
   4f5d8:	bf 00 00 00 00       	mov    $0x0,%edi	4f5d9: R_X86_64_32	.bss+0x1fa8
   4f5dd:	e8 00 00 00 00       	call   4f5e2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f72>	4f5de: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f5e2:	e9 16 01 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f5e7:	48 89 c3             	mov    %rax,%rbx
   4f5ea:	bf 00 00 00 00       	mov    $0x0,%edi	4f5eb: R_X86_64_32	.bss+0x1f88
   4f5ef:	e8 00 00 00 00       	call   4f5f4 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f84>	4f5f0: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f5f4:	e9 04 01 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f5f9:	48 89 c3             	mov    %rax,%rbx
   4f5fc:	bf 00 00 00 00       	mov    $0x0,%edi	4f5fd: R_X86_64_32	.bss+0x1f28
   4f601:	e8 00 00 00 00       	call   4f606 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2f96>	4f602: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f606:	e9 f2 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f60b:	48 89 c3             	mov    %rax,%rbx
   4f60e:	bf 00 00 00 00       	mov    $0x0,%edi	4f60f: R_X86_64_32	.bss+0x1ee8
   4f613:	e8 00 00 00 00       	call   4f618 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fa8>	4f614: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f618:	e9 e0 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f61d:	48 89 c3             	mov    %rax,%rbx
   4f620:	bf 00 00 00 00       	mov    $0x0,%edi	4f621: R_X86_64_32	.bss+0x1ec8
   4f625:	e8 00 00 00 00       	call   4f62a <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fba>	4f626: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f62a:	e9 ce 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f62f:	48 89 c3             	mov    %rax,%rbx
   4f632:	bf 00 00 00 00       	mov    $0x0,%edi	4f633: R_X86_64_32	.bss+0x1f08
   4f637:	e8 00 00 00 00       	call   4f63c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fcc>	4f638: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f63c:	e9 bc 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f641:	48 89 c3             	mov    %rax,%rbx
   4f644:	bf 00 00 00 00       	mov    $0x0,%edi	4f645: R_X86_64_32	.bss+0x1f98
   4f649:	e8 00 00 00 00       	call   4f64e <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2fde>	4f64a: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f64e:	e9 aa 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f653:	48 89 c3             	mov    %rax,%rbx
   4f656:	bf 00 00 00 00       	mov    $0x0,%edi	4f657: R_X86_64_32	.bss+0x1f38
   4f65b:	e8 00 00 00 00       	call   4f660 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x2ff0>	4f65c: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f660:	e9 98 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f665:	48 89 c3             	mov    %rax,%rbx
   4f668:	bf 00 00 00 00       	mov    $0x0,%edi	4f669: R_X86_64_32	.bss+0x1f78
   4f66d:	e8 00 00 00 00       	call   4f672 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3002>	4f66e: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f672:	e9 86 00 00 00       	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f677:	48 89 c3             	mov    %rax,%rbx
   4f67a:	bf 00 00 00 00       	mov    $0x0,%edi	4f67b: R_X86_64_32	.bss+0x1f18
   4f67f:	e8 00 00 00 00       	call   4f684 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3014>	4f680: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f684:	eb 77                	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f686:	48 89 c3             	mov    %rax,%rbx
   4f689:	bf 00 00 00 00       	mov    $0x0,%edi	4f68a: R_X86_64_32	.bss+0x1ed8
   4f68e:	e8 00 00 00 00       	call   4f693 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3023>	4f68f: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f693:	eb 68                	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f695:	48 89 c3             	mov    %rax,%rbx
   4f698:	bf 00 00 00 00       	mov    $0x0,%edi	4f699: R_X86_64_32	.bss+0x1eb8
   4f69d:	e8 00 00 00 00       	call   4f6a2 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3032>	4f69e: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f6a2:	eb 59                	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f6a4:	48 89 c3             	mov    %rax,%rbx
   4f6a7:	bf 00 00 00 00       	mov    $0x0,%edi	4f6a8: R_X86_64_32	.bss+0x1ef8
   4f6ac:	e8 00 00 00 00       	call   4f6b1 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3041>	4f6ad: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f6b1:	eb 4a                	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f6b3:	48 89 c3             	mov    %rax,%rbx
   4f6b6:	bf 00 00 00 00       	mov    $0x0,%edi	4f6b7: R_X86_64_32	.bss+0x1fb8
   4f6bb:	e8 00 00 00 00       	call   4f6c0 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3050>	4f6bc: R_X86_64_PLT32	__cxa_guard_abort-0x4
   4f6c0:	eb 3b                	jmp    4f6fd <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308d>
   4f6c2:	eb 36                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6c4:	eb 34                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6c6:	eb 32                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6c8:	eb 30                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6ca:	eb 2e                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6cc:	eb 2c                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6ce:	eb 2a                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6d0:	eb 4a                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6d2:	eb 48                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6d4:	eb 46                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6d6:	eb 44                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6d8:	eb 42                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6da:	eb 40                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6dc:	eb 3e                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6de:	eb 1a                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6e0:	eb 18                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6e2:	eb 16                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6e4:	eb 14                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6e6:	eb 12                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6e8:	eb 10                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6ea:	eb 0e                	jmp    4f6fa <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x308a>
   4f6ec:	eb 2e                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6ee:	eb 2c                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6f0:	eb 2a                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6f2:	eb 28                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6f4:	eb 26                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6f6:	eb 24                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6f8:	eb 22                	jmp    4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>
   4f6fa:	48 89 c3             	mov    %rax,%rbx
   4f6fd:	48 8d 7c 24 50       	lea    0x50(%rsp),%rdi
   4f702:	e8 00 00 00 00       	call   4f707 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x3097>	4f703: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f707:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4f70f:	e8 00 00 00 00       	call   4f714 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30a4>	4f710: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f714:	48 89 df             	mov    %rbx,%rdi
   4f717:	e8 00 00 00 00       	call   4f71c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30ac>	4f718: R_X86_64_PLT32	_Unwind_Resume-0x4
   4f71c:	48 89 c3             	mov    %rax,%rbx
   4f71f:	48 8d bc 24 88 00 00 00 	lea    0x88(%rsp),%rdi
   4f727:	e8 00 00 00 00       	call   4f72c <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30bc>	4f728: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   4f72c:	48 89 df             	mov    %rbx,%rdi
   4f72f:	e8 00 00 00 00       	call   4f734 <_ZN6strata7kernels12_GLOBAL__N_115launch_q6k_wideILi6EEEvPKvS4_PfiiPN4sycl3_V15queueE+0x30c4>	4f730: R_X86_64_PLT32	_Unwind_Resume-0x4

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
