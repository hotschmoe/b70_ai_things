
/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000067f90 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv>:
   67f90:	55                   	push   %rbp
   67f91:	41 57                	push   %r15
   67f93:	41 56                	push   %r14
   67f95:	41 55                	push   %r13
   67f97:	41 54                	push   %r12
   67f99:	53                   	push   %rbx
   67f9a:	48 81 ec c8 01 00 00 	sub    $0x1c8,%rsp
   67fa1:	44 89 cd             	mov    %r9d,%ebp
   67fa4:	45 89 c5             	mov    %r8d,%r13d
   67fa7:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
   67fac:	49 89 d7             	mov    %rdx,%r15
   67faf:	49 89 f4             	mov    %rsi,%r12
   67fb2:	89 fb                	mov    %edi,%ebx
   67fb4:	48 8b 84 24 08 02 00 00 	mov    0x208(%rsp),%rax
   67fbc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   67fc1:	44 8b b4 24 00 02 00 00 	mov    0x200(%rsp),%r14d
   67fc9:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 67fd0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x40>	67fcc: R_X86_64_PC32	.bss+0x54
   67fd0:	84 c0                	test   %al,%al
   67fd2:	0f 84 4b 07 00 00    	je     68723 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x793>
   67fd8:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 67fdf <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x4f>	67fda: R_X86_64_PC32	.bss+0x4b
   67fdf:	0f 84 c0 00 00 00    	je     680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   67fe5:	83 fb 0d             	cmp    $0xd,%ebx
   67fe8:	0f 85 b7 00 00 00    	jne    680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   67fee:	41 81 fd 00 0a 00 00 	cmp    $0xa00,%r13d
   67ff5:	0f 85 aa 00 00 00    	jne    680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   67ffb:	81 fd 00 ca 03 00    	cmp    $0x3ca00,%ebp
   68001:	0f 85 9e 00 00 00    	jne    680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   68007:	44 89 f0             	mov    %r14d,%eax
   6800a:	83 c8 04             	or     $0x4,%eax
   6800d:	83 f8 06             	cmp    $0x6,%eax
   68010:	0f 85 8f 00 00 00    	jne    680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   68016:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 6801d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8d>	68019: R_X86_64_PC32	.bss+0x344
   6801d:	84 c0                	test   %al,%al
   6801f:	0f 84 df 07 00 00    	je     68804 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x874>
   68025:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 6802c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9c>	68027: R_X86_64_PC32	.bss+0x33b
   6802c:	74 77                	je     680a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x115>
   6802e:	4d 85 e4             	test   %r12,%r12
   68031:	0f 84 32 07 00 00    	je     68769 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7d9>
   68037:	41 f6 c4 03          	test   $0x3,%r12b
   6803b:	0f 85 28 07 00 00    	jne    68769 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7d9>
   68041:	4d 85 ff             	test   %r15,%r15
   68044:	4c 8b 4c 24 20       	mov    0x20(%rsp),%r9
   68049:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
   6804e:	0f 84 34 07 00 00    	je     68788 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7f8>
   68054:	41 f6 c7 03          	test   $0x3,%r15b
   68058:	0f 85 2a 07 00 00    	jne    68788 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7f8>
   6805e:	48 85 d2             	test   %rdx,%rdx
   68061:	0f 84 40 07 00 00    	je     687a7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x817>
   68067:	f6 c2 03             	test   $0x3,%dl
   6806a:	0f 85 37 07 00 00    	jne    687a7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x817>
   68070:	4d 85 c9             	test   %r9,%r9
   68073:	0f 84 02 08 00 00    	je     6887b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8eb>
   68079:	41 83 fe 02          	cmp    $0x2,%r14d
   6807d:	0f 85 45 06 00 00    	jne    686c8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x738>
   68083:	c7 04 24 02 00 00 00 	movl   $0x2,(%rsp)
   6808a:	4c 89 e7             	mov    %r12,%rdi
   6808d:	4c 89 fe             	mov    %r15,%rsi
   68090:	b9 00 0a 00 00       	mov    $0xa00,%ecx
   68095:	41 b8 00 ca 03 00    	mov    $0x3ca00,%r8d
   6809b:	e8 f0 0b 00 00       	call   68c90 <_ZN6strata7kernels12_GLOBAL__N_111launch_wideINS1_7WideQ5KELi2EEEvPKvS5_PfiiPN4sycl3_V15queueEi>
   680a0:	e9 62 06 00 00       	jmp    68707 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x777>
   680a5:	89 e8                	mov    %ebp,%eax
   680a7:	35 00 ca 03 00       	xor    $0x3ca00,%eax
   680ac:	44 89 e9             	mov    %r13d,%ecx
   680af:	81 f1 00 0a 00 00    	xor    $0xa00,%ecx
   680b5:	89 da                	mov    %ebx,%edx
   680b7:	83 f2 0d             	xor    $0xd,%edx
   680ba:	09 ca                	or     %ecx,%edx
   680bc:	09 c2                	or     %eax,%edx
   680be:	0f 94 c0             	sete   %al
   680c1:	0f 85 c1 02 00 00    	jne    68388 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x3f8>
   680c7:	41 83 fe 02          	cmp    $0x2,%r14d
   680cb:	0f 85 b7 02 00 00    	jne    68388 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x3f8>
   680d1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 680d8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x148>	680d4: R_X86_64_PC32	.bss+0x344
   680d8:	84 c0                	test   %al,%al
   680da:	0f 84 e6 06 00 00    	je     687c6 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x836>
   680e0:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 680e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x157>	680e2: R_X86_64_PC32	.bss+0x33b
   680e7:	0f 84 b2 05 00 00    	je     6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   680ed:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 680f4 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x164>	680f0: R_X86_64_PC32	.bss+0x354
   680f4:	84 c0                	test   %al,%al
   680f6:	0f 84 46 07 00 00    	je     68842 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8b2>
   680fc:	83 3d 00 00 00 00 01 	cmpl   $0x1,0x0(%rip)        # 68103 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x173>	680fe: R_X86_64_PC32	.bss+0x34b
   68103:	0f 85 96 05 00 00    	jne    6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   68109:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 68110 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x180>	6810c: R_X86_64_PC32	.bss+0x364
   68110:	84 c0                	test   %al,%al
   68112:	0f 84 56 08 00 00    	je     6896e <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9de>
   68118:	44 8b 35 00 00 00 00 	mov    0x0(%rip),%r14d        # 6811f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x18f>	6811b: R_X86_64_PC32	.bss+0x35c
   6811f:	45 85 f6             	test   %r14d,%r14d
   68122:	0f 84 77 05 00 00    	je     6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   68128:	4d 85 e4             	test   %r12,%r12
   6812b:	0f 84 a7 07 00 00    	je     688d8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x948>
   68131:	41 f6 c4 03          	test   $0x3,%r12b
   68135:	0f 85 9d 07 00 00    	jne    688d8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x948>
   6813b:	4d 85 ff             	test   %r15,%r15
   6813e:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
   68143:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
   68148:	0f 84 a9 07 00 00    	je     688f7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x967>
   6814e:	41 f6 c7 03          	test   $0x3,%r15b
   68152:	0f 85 9f 07 00 00    	jne    688f7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x967>
   68158:	48 85 db             	test   %rbx,%rbx
   6815b:	0f 84 b5 07 00 00    	je     68916 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x986>
   68161:	f6 c3 03             	test   $0x3,%bl
   68164:	0f 85 ac 07 00 00    	jne    68916 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x986>
   6816a:	4d 85 ed             	test   %r13,%r13
   6816d:	0f 84 b2 08 00 00    	je     68a25 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa95>
   68173:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 6817a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x1ea>	68176: R_X86_64_PC32	.bss+0x64
   6817a:	84 c0                	test   %al,%al
   6817c:	0f 84 c2 08 00 00    	je     68a44 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xab4>
   68182:	4c 89 a4 24 c8 00 00 00 	mov    %r12,0xc8(%rsp)
   6818a:	4c 89 bc 24 d0 00 00 00 	mov    %r15,0xd0(%rsp)
   68192:	48 89 9c 24 d8 00 00 00 	mov    %rbx,0xd8(%rsp)
   6819a:	48 b8 00 0a 00 00 00 ca 03 00 	movabs $0x3ca0000000a00,%rax
   681a4:	48 89 84 24 e0 00 00 00 	mov    %rax,0xe0(%rsp)
   681ac:	44 89 b4 24 e8 00 00 00 	mov    %r14d,0xe8(%rsp)
   681b4:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 681bb <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x22b>	681b7: R_X86_64_PC32	.rodata+0x78b6c
   681bb:	48 89 84 24 00 01 00 00 	mov    %rax,0x100(%rsp)
   681c3:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 681ca <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x23a>	681c6: R_X86_64_PC32	.rodata+0x78b5c
   681ca:	0f 29 84 24 f0 00 00 00 	movaps %xmm0,0xf0(%rsp)
   681d2:	48 8d 5c 24 40       	lea    0x40(%rsp),%rbx
   681d7:	48 8d b4 24 f0 00 00 00 	lea    0xf0(%rsp),%rsi
   681df:	48 89 df             	mov    %rbx,%rdi
   681e2:	e8 00 00 00 00       	call   681e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x257>	681e3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   681e7:	48 89 df             	mov    %rbx,%rdi
   681ea:	e8 00 00 00 00       	call   681ef <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x25f>	681eb: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   681ef:	48 89 c3             	mov    %rax,%rbx
   681f2:	48 c7 84 24 80 00 00 00 01 00 00 00 	movq   $0x1,0x80(%rsp)
   681fe:	48 c7 84 24 88 00 00 00 04 00 00 00 	movq   $0x4,0x88(%rsp)
   6820a:	48 c7 84 24 90 00 00 00 00 50 1e 00 	movq   $0x1e5000,0x90(%rsp)
   68216:	48 c7 84 24 98 00 00 00 01 00 00 00 	movq   $0x1,0x98(%rsp)
   68222:	48 c7 84 24 a0 00 00 00 04 00 00 00 	movq   $0x4,0xa0(%rsp)
   6822e:	48 c7 84 24 a8 00 00 00 20 00 00 00 	movq   $0x20,0xa8(%rsp)
   6823a:	0f 57 c0             	xorps  %xmm0,%xmm0
   6823d:	0f 11 84 24 b0 00 00 00 	movups %xmm0,0xb0(%rsp)
   68245:	48 c7 84 24 c0 00 00 00 00 00 00 00 	movq   $0x0,0xc0(%rsp)
   68251:	4c 8d b4 24 80 00 00 00 	lea    0x80(%rsp),%r14
   68259:	4c 89 f7             	mov    %r14,%rdi
   6825c:	e8 00 00 00 00       	call   68261 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x2d1>	6825d: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   68261:	48 8d 84 24 98 00 00 00 	lea    0x98(%rsp),%rax
   68269:	48 8d 8c 24 b0 00 00 00 	lea    0xb0(%rsp),%rcx
   68271:	4c 89 74 24 60       	mov    %r14,0x60(%rsp)
   68276:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
   6827b:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
   68280:	48 c7 44 24 78 03 00 00 00 	movq   $0x3,0x78(%rsp)
   68289:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   6828e:	48 89 de             	mov    %rbx,%rsi
   68291:	e8 00 00 00 00       	call   68296 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x306>	68292: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   68296:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 6829d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x30d>	68299: R_X86_64_PC32	.bss+0x374
   6829d:	84 c0                	test   %al,%al
   6829f:	0f 84 e2 07 00 00    	je     68a87 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xaf7>
   682a5:	48 8b 1d 00 00 00 00 	mov    0x0(%rip),%rbx        # 682ac <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x31c>	682a8: R_X86_64_PC32	.bss+0x36c
   682ac:	c6 84 24 38 01 00 00 00 	movb   $0x0,0x138(%rsp)
   682b4:	c6 84 24 42 01 00 00 00 	movb   $0x0,0x142(%rsp)
   682bc:	c6 84 24 44 01 00 00 00 	movb   $0x0,0x144(%rsp)
   682c4:	c6 84 24 4c 01 00 00 00 	movb   $0x0,0x14c(%rsp)
   682cc:	c6 84 24 54 01 00 00 00 	movb   $0x0,0x154(%rsp)
   682d4:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   682dc:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   682e4:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   682ec:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   682f4:	c6 84 24 80 01 00 00 00 	movb   $0x0,0x180(%rsp)
   682fc:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   68304:	c6 84 24 b8 01 00 00 00 	movb   $0x0,0x1b8(%rsp)
   6830c:	c6 84 24 c0 01 00 00 01 	movb   $0x1,0x1c0(%rsp)
   68314:	48 c7 44 24 50 00 00 00 00 	movq   $0x0,0x50(%rsp)	68319: R_X86_64_32S	.rodata+0x10e70
   6831d:	48 8d 84 24 c8 00 00 00 	lea    0xc8(%rsp),%rax
   68325:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
   6832a:	0f 57 c0             	xorps  %xmm0,%xmm0
   6832d:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   68335:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   6833a:	e8 00 00 00 00       	call   6833f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x3af>	6833b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   6833f:	31 c9                	xor    %ecx,%ecx
   68341:	80 7c 24 38 00       	cmpb   $0x0,0x38(%rsp)
   68346:	0f 94 c1             	sete   %cl
   68349:	0f 28 84 24 10 01 00 00 	movaps 0x110(%rsp),%xmm0
   68351:	0f 11 04 24          	movups %xmm0,(%rsp)
   68355:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   68359:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   6835e:	48 8d bc 24 20 01 00 00 	lea    0x120(%rsp),%rdi
   68366:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
   6836b:	48 8d 4c 24 50       	lea    0x50(%rsp),%rcx
   68370:	4c 8d 8c 24 30 01 00 00 	lea    0x130(%rsp),%r9
   68378:	4c 89 ee             	mov    %r13,%rsi
   6837b:	49 89 d8             	mov    %rbx,%r8
   6837e:	e8 00 00 00 00       	call   68383 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x3f3>	6837f: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   68383:	e9 c3 02 00 00       	jmp    6864b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x6bb>
   68388:	41 83 fe 06          	cmp    $0x6,%r14d
   6838c:	0f 95 c1             	setne  %cl
   6838f:	34 01                	xor    $0x1,%al
   68391:	08 c8                	or     %cl,%al
   68393:	0f 85 06 03 00 00    	jne    6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   68399:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 683a0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x410>	6839c: R_X86_64_PC32	.bss+0x344
   683a0:	84 c0                	test   %al,%al
   683a2:	0f 84 f2 04 00 00    	je     6889a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x90a>
   683a8:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 683af <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x41f>	683aa: R_X86_64_PC32	.bss+0x33b
   683af:	0f 84 ea 02 00 00    	je     6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   683b5:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 683bc <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x42c>	683b8: R_X86_64_PC32	.bss+0x354
   683bc:	84 c0                	test   %al,%al
   683be:	0f 84 71 05 00 00    	je     68935 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9a5>
   683c4:	83 3d 00 00 00 00 01 	cmpl   $0x1,0x0(%rip)        # 683cb <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x43b>	683c6: R_X86_64_PC32	.bss+0x34b
   683cb:	0f 85 ce 02 00 00    	jne    6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   683d1:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 683d8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x448>	683d4: R_X86_64_PC32	.bss+0x384
   683d8:	84 c0                	test   %al,%al
   683da:	0f 84 de 06 00 00    	je     68abe <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb2e>
   683e0:	44 8b 35 00 00 00 00 	mov    0x0(%rip),%r14d        # 683e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x457>	683e3: R_X86_64_PC32	.bss+0x37c
   683e7:	45 85 f6             	test   %r14d,%r14d
   683ea:	0f 84 af 02 00 00    	je     6869f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x70f>
   683f0:	4d 85 e4             	test   %r12,%r12
   683f3:	0f 84 cf 05 00 00    	je     689c8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa38>
   683f9:	41 f6 c4 03          	test   $0x3,%r12b
   683fd:	0f 85 c5 05 00 00    	jne    689c8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa38>
   68403:	4d 85 ff             	test   %r15,%r15
   68406:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
   6840b:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
   68410:	0f 84 d1 05 00 00    	je     689e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa57>
   68416:	41 f6 c7 03          	test   $0x3,%r15b
   6841a:	0f 85 c7 05 00 00    	jne    689e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa57>
   68420:	48 85 db             	test   %rbx,%rbx
   68423:	0f 84 dd 05 00 00    	je     68a06 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa76>
   68429:	f6 c3 03             	test   $0x3,%bl
   6842c:	0f 85 d4 05 00 00    	jne    68a06 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa76>
   68432:	4d 85 ed             	test   %r13,%r13
   68435:	0f 84 e0 06 00 00    	je     68b1b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb8b>
   6843b:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 68442 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x4b2>	6843e: R_X86_64_PC32	.bss+0x74
   68442:	84 c0                	test   %al,%al
   68444:	0f 84 fd 06 00 00    	je     68b47 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbb7>
   6844a:	4c 89 a4 24 c8 00 00 00 	mov    %r12,0xc8(%rsp)
   68452:	4c 89 bc 24 d0 00 00 00 	mov    %r15,0xd0(%rsp)
   6845a:	48 89 9c 24 d8 00 00 00 	mov    %rbx,0xd8(%rsp)
   68462:	48 b8 00 0a 00 00 00 ca 03 00 	movabs $0x3ca0000000a00,%rax
   6846c:	48 89 84 24 e0 00 00 00 	mov    %rax,0xe0(%rsp)
   68474:	44 89 b4 24 e8 00 00 00 	mov    %r14d,0xe8(%rsp)
   6847c:	48 8b 05 00 00 00 00 	mov    0x0(%rip),%rax        # 68483 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x4f3>	6847f: R_X86_64_PC32	.rodata+0x78b6c
   68483:	48 89 84 24 00 01 00 00 	mov    %rax,0x100(%rsp)
   6848b:	0f 10 05 00 00 00 00 	movups 0x0(%rip),%xmm0        # 68492 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x502>	6848e: R_X86_64_PC32	.rodata+0x78b5c
   68492:	0f 29 84 24 f0 00 00 00 	movaps %xmm0,0xf0(%rsp)
   6849a:	48 8d 5c 24 40       	lea    0x40(%rsp),%rbx
   6849f:	48 8d b4 24 f0 00 00 00 	lea    0xf0(%rsp),%rsi
   684a7:	48 89 df             	mov    %rbx,%rdi
   684aa:	e8 00 00 00 00       	call   684af <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x51f>	684ab: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   684af:	48 89 df             	mov    %rbx,%rdi
   684b2:	e8 00 00 00 00       	call   684b7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x527>	684b3: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   684b7:	48 89 c3             	mov    %rax,%rbx
   684ba:	48 c7 84 24 80 00 00 00 01 00 00 00 	movq   $0x1,0x80(%rsp)
   684c6:	48 c7 84 24 88 00 00 00 04 00 00 00 	movq   $0x4,0x88(%rsp)
   684d2:	48 c7 84 24 90 00 00 00 00 50 1e 00 	movq   $0x1e5000,0x90(%rsp)
   684de:	48 c7 84 24 98 00 00 00 01 00 00 00 	movq   $0x1,0x98(%rsp)
   684ea:	48 c7 84 24 a0 00 00 00 04 00 00 00 	movq   $0x4,0xa0(%rsp)
   684f6:	48 c7 84 24 a8 00 00 00 20 00 00 00 	movq   $0x20,0xa8(%rsp)
   68502:	0f 57 c0             	xorps  %xmm0,%xmm0
   68505:	0f 11 84 24 b0 00 00 00 	movups %xmm0,0xb0(%rsp)
   6850d:	48 c7 84 24 c0 00 00 00 00 00 00 00 	movq   $0x0,0xc0(%rsp)
   68519:	4c 8d b4 24 80 00 00 00 	lea    0x80(%rsp),%r14
   68521:	4c 89 f7             	mov    %r14,%rdi
   68524:	e8 00 00 00 00       	call   68529 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x599>	68525: R_X86_64_PLT32	_ZN4sycl3_V16detail15checkValueRangeILi3EEEvRKNS0_8nd_rangeIXT_EEE-0x4
   68529:	48 8d 84 24 98 00 00 00 	lea    0x98(%rsp),%rax
   68531:	48 8d 8c 24 b0 00 00 00 	lea    0xb0(%rsp),%rcx
   68539:	4c 89 74 24 60       	mov    %r14,0x60(%rsp)
   6853e:	48 89 44 24 68       	mov    %rax,0x68(%rsp)
   68543:	48 89 4c 24 70       	mov    %rcx,0x70(%rsp)
   68548:	48 c7 44 24 78 03 00 00 00 	movq   $0x3,0x78(%rsp)
   68551:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   68556:	48 89 de             	mov    %rbx,%rsi
   68559:	e8 00 00 00 00       	call   6855e <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x5ce>	6855a: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tC1ERKNS1_13code_locationE-0x4
   6855e:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 68565 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x5d5>	68561: R_X86_64_PC32	.bss+0x394
   68565:	84 c0                	test   %al,%al
   68567:	0f 84 1d 06 00 00    	je     68b8a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbfa>
   6856d:	48 8b 1d 00 00 00 00 	mov    0x0(%rip),%rbx        # 68574 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x5e4>	68570: R_X86_64_PC32	.bss+0x38c
   68574:	c6 84 24 38 01 00 00 00 	movb   $0x0,0x138(%rsp)
   6857c:	c6 84 24 42 01 00 00 00 	movb   $0x0,0x142(%rsp)
   68584:	c6 84 24 44 01 00 00 00 	movb   $0x0,0x144(%rsp)
   6858c:	c6 84 24 4c 01 00 00 00 	movb   $0x0,0x14c(%rsp)
   68594:	c6 84 24 54 01 00 00 00 	movb   $0x0,0x154(%rsp)
   6859c:	c6 84 24 5c 01 00 00 00 	movb   $0x0,0x15c(%rsp)
   685a4:	c6 84 24 64 01 00 00 00 	movb   $0x0,0x164(%rsp)
   685ac:	c6 84 24 6c 01 00 00 00 	movb   $0x0,0x16c(%rsp)
   685b4:	c6 84 24 74 01 00 00 00 	movb   $0x0,0x174(%rsp)
   685bc:	c6 84 24 80 01 00 00 00 	movb   $0x0,0x180(%rsp)
   685c4:	c6 84 24 98 01 00 00 00 	movb   $0x0,0x198(%rsp)
   685cc:	c6 84 24 b8 01 00 00 00 	movb   $0x0,0x1b8(%rsp)
   685d4:	c6 84 24 c0 01 00 00 01 	movb   $0x1,0x1c0(%rsp)
   685dc:	48 c7 44 24 50 00 00 00 00 	movq   $0x0,0x50(%rsp)	685e1: R_X86_64_32S	.rodata+0x11088
   685e5:	48 8d 84 24 c8 00 00 00 	lea    0xc8(%rsp),%rax
   685ed:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
   685f2:	0f 57 c0             	xorps  %xmm0,%xmm0
   685f5:	0f 29 84 24 10 01 00 00 	movaps %xmm0,0x110(%rsp)
   685fd:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   68602:	e8 00 00 00 00       	call   68607 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x677>	68603: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_t5queryEv-0x4
   68607:	31 c9                	xor    %ecx,%ecx
   68609:	80 7c 24 38 00       	cmpb   $0x0,0x38(%rsp)
   6860e:	0f 94 c1             	sete   %cl
   68611:	0f 28 84 24 10 01 00 00 	movaps 0x110(%rsp),%xmm0
   68619:	0f 11 04 24          	movups %xmm0,(%rsp)
   6861d:	89 4c 24 18          	mov    %ecx,0x18(%rsp)
   68621:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
   68626:	48 8d bc 24 20 01 00 00 	lea    0x120(%rsp),%rdi
   6862e:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
   68633:	48 8d 4c 24 50       	lea    0x50(%rsp),%rcx
   68638:	4c 8d 8c 24 30 01 00 00 	lea    0x130(%rsp),%r9
   68640:	4c 89 ee             	mov    %r13,%rsi
   68643:	49 89 d8             	mov    %rbx,%r8
   68646:	e8 00 00 00 00       	call   6864b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x6bb>	68647: R_X86_64_PLT32	_ZN4sycl3_V136submit_kernel_direct_with_event_implERKNS0_5queueERKNS0_6detail16nd_range_view_v113nd_range_viewERNS4_17HostKernelRefBaseEPNS4_16DeviceKernelInfoENS0_4spanIKNS0_5eventELm18446744073709551615EEERKNS4_27kernel_launch_properties_v111PropsHolderIJNS0_3ext6oneapi12experimental23work_group_scratch_sizeENSJ_5intel12experimental12cache_configENSL_17use_root_sync_keyENSL_23work_group_progress_keyENSL_22sub_group_progress_keyENSL_22work_item_progress_keyENSL_4cuda12cluster_sizeILi1EEENSV_ILi2EEENSV_ILi3EEEEEERKNS4_13code_locationEb-0x4
   6864b:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   68650:	e8 00 00 00 00       	call   68655 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x6c5>	68651: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   68655:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
   6865a:	e8 00 00 00 00       	call   6865f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x6cf>	6865b: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   6865f:	48 8b 9c 24 28 01 00 00 	mov    0x128(%rsp),%rbx
   68667:	48 85 db             	test   %rbx,%rbx
   6866a:	0f 84 97 00 00 00    	je     68707 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x777>
   68670:	48 8b 43 08          	mov    0x8(%rbx),%rax
   68674:	48 b9 01 00 00 00 01 00 00 00 	movabs $0x100000001,%rcx
   6867e:	48 39 c8             	cmp    %rcx,%rax
   68681:	75 64                	jne    686e7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x757>
   68683:	48 c7 43 08 00 00 00 00 	movq   $0x0,0x8(%rbx)
   6868b:	48 8b 03             	mov    (%rbx),%rax
   6868e:	48 89 df             	mov    %rbx,%rdi
   68691:	ff 50 10             	call   *0x10(%rax)
   68694:	48 8b 03             	mov    (%rbx),%rax
   68697:	48 89 df             	mov    %rbx,%rdi
   6869a:	ff 50 18             	call   *0x18(%rax)
   6869d:	eb 68                	jmp    68707 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x777>
   6869f:	89 df                	mov    %ebx,%edi
   686a1:	4c 89 e6             	mov    %r12,%rsi
   686a4:	4c 89 fa             	mov    %r15,%rdx
   686a7:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
   686ac:	45 89 e8             	mov    %r13d,%r8d
   686af:	41 89 e9             	mov    %ebp,%r9d
   686b2:	48 81 c4 c8 01 00 00 	add    $0x1c8,%rsp
   686b9:	5b                   	pop    %rbx
   686ba:	41 5c                	pop    %r12
   686bc:	41 5d                	pop    %r13
   686be:	41 5e                	pop    %r14
   686c0:	41 5f                	pop    %r15
   686c2:	5d                   	pop    %rbp
   686c3:	e9 00 00 00 00       	jmp    686c8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x738>	686c4: R_X86_64_PLT32	_ZN6strata7kernels11native_mmvqEiPKvS2_PfiiiPv-0x4
   686c8:	c7 04 24 02 00 00 00 	movl   $0x2,(%rsp)
   686cf:	4c 89 e7             	mov    %r12,%rdi
   686d2:	4c 89 fe             	mov    %r15,%rsi
   686d5:	b9 00 0a 00 00       	mov    $0xa00,%ecx
   686da:	41 b8 00 ca 03 00    	mov    $0x3ca00,%r8d
   686e0:	e8 9b 0d 00 00       	call   69480 <_ZN6strata7kernels12_GLOBAL__N_111launch_wideINS1_7WideQ5KELi6EEEvPKvS5_PfiiPN4sycl3_V15queueEi>
   686e5:	eb 20                	jmp    68707 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x777>
   686e7:	80 3d 00 00 00 00 00 	cmpb   $0x0,0x0(%rip)        # 686ee <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x75e>	686e9: R_X86_64_PC32	__libc_single_threaded-0x5
   686ee:	74 08                	je     686f8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x768>
   686f0:	8d 48 ff             	lea    -0x1(%rax),%ecx
   686f3:	89 4b 08             	mov    %ecx,0x8(%rbx)
   686f6:	eb 0a                	jmp    68702 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x772>
   686f8:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
   686fd:	f0 0f c1 43 08       	lock xadd %eax,0x8(%rbx)
   68702:	83 f8 01             	cmp    $0x1,%eax
   68705:	74 12                	je     68719 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x789>
   68707:	48 81 c4 c8 01 00 00 	add    $0x1c8,%rsp
   6870e:	5b                   	pop    %rbx
   6870f:	41 5c                	pop    %r12
   68711:	41 5d                	pop    %r13
   68713:	41 5e                	pop    %r14
   68715:	41 5f                	pop    %r15
   68717:	5d                   	pop    %rbp
   68718:	c3                   	ret
   68719:	48 89 df             	mov    %rbx,%rdi
   6871c:	e8 00 00 00 00       	call   68721 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x791>	6871d: R_X86_64_PLT32	_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv-0x4
   68721:	eb e4                	jmp    68707 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x777>
   68723:	bf 00 00 00 00       	mov    $0x0,%edi	68724: R_X86_64_32	.bss+0x58
   68728:	e8 00 00 00 00       	call   6872d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x79d>	68729: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   6872d:	85 c0                	test   %eax,%eax
   6872f:	0f 84 a3 f8 ff ff    	je     67fd8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x48>
   68735:	bf 00 00 00 00       	mov    $0x0,%edi	68736: R_X86_64_32	.rodata.str1.1+0x29ac
   6873a:	e8 00 00 00 00       	call   6873f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7af>	6873b: R_X86_64_PLT32	getenv-0x4
   6873f:	48 85 c0             	test   %rax,%rax
   68742:	74 0e                	je     68752 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7c2>
   68744:	80 38 31             	cmpb   $0x31,(%rax)
   68747:	75 09                	jne    68752 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7c2>
   68749:	80 78 01 00          	cmpb   $0x0,0x1(%rax)
   6874d:	0f 94 c0             	sete   %al
   68750:	eb 02                	jmp    68754 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7c4>
   68752:	31 c0                	xor    %eax,%eax
   68754:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 6875a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7ca>	68756: R_X86_64_PC32	.bss+0x4c
   6875a:	bf 00 00 00 00       	mov    $0x0,%edi	6875b: R_X86_64_32	.bss+0x58
   6875f:	e8 00 00 00 00       	call   68764 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7d4>	68760: R_X86_64_PLT32	__cxa_guard_release-0x4
   68764:	e9 6f f8 ff ff       	jmp    67fd8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x48>
   68769:	bf 10 00 00 00       	mov    $0x10,%edi
   6876e:	e8 00 00 00 00       	call   68773 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7e3>	6876f: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68773:	49 89 c6             	mov    %rax,%r14
   68776:	be 00 00 00 00       	mov    $0x0,%esi	68777: R_X86_64_32	.rodata.str1.1+0x2cb
   6877b:	48 89 c7             	mov    %rax,%rdi
   6877e:	e8 00 00 00 00       	call   68783 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x7f3>	6877f: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68783:	e9 ad 03 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68788:	bf 10 00 00 00       	mov    $0x10,%edi
   6878d:	e8 00 00 00 00       	call   68792 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x802>	6878e: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68792:	49 89 c6             	mov    %rax,%r14
   68795:	be 00 00 00 00       	mov    $0x0,%esi	68796: R_X86_64_32	.rodata.str1.1+0x2cb
   6879a:	48 89 c7             	mov    %rax,%rdi
   6879d:	e8 00 00 00 00       	call   687a2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x812>	6879e: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   687a2:	e9 8e 03 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   687a7:	bf 10 00 00 00       	mov    $0x10,%edi
   687ac:	e8 00 00 00 00       	call   687b1 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x821>	687ad: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   687b1:	49 89 c6             	mov    %rax,%r14
   687b4:	be 00 00 00 00       	mov    $0x0,%esi	687b5: R_X86_64_32	.rodata.str1.1+0x2cb
   687b9:	48 89 c7             	mov    %rax,%rdi
   687bc:	e8 00 00 00 00       	call   687c1 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x831>	687bd: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   687c1:	e9 6f 03 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   687c6:	bf 00 00 00 00       	mov    $0x0,%edi	687c7: R_X86_64_32	.bss+0x348
   687cb:	e8 00 00 00 00       	call   687d0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x840>	687cc: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   687d0:	85 c0                	test   %eax,%eax
   687d2:	0f 84 08 f9 ff ff    	je     680e0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x150>
   687d8:	bf 00 00 00 00       	mov    $0x0,%edi	687d9: R_X86_64_32	.rodata.str1.1+0x29c1
   687dd:	e8 00 00 00 00       	call   687e2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x852>	687de: R_X86_64_PLT32	getenv-0x4
   687e2:	48 85 c0             	test   %rax,%rax
   687e5:	0f 84 af 01 00 00    	je     6899a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa0a>
   687eb:	48 89 c7             	mov    %rax,%rdi
   687ee:	31 f6                	xor    %esi,%esi
   687f0:	ba 0a 00 00 00       	mov    $0xa,%edx
   687f5:	e8 00 00 00 00       	call   687fa <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x86a>	687f6: R_X86_64_PLT32	__isoc23_strtol-0x4
   687fa:	85 c0                	test   %eax,%eax
   687fc:	0f 95 c0             	setne  %al
   687ff:	e9 98 01 00 00       	jmp    6899c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa0c>
   68804:	bf 00 00 00 00       	mov    $0x0,%edi	68805: R_X86_64_32	.bss+0x348
   68809:	e8 00 00 00 00       	call   6880e <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x87e>	6880a: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   6880e:	85 c0                	test   %eax,%eax
   68810:	0f 84 0f f8 ff ff    	je     68025 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x95>
   68816:	bf 00 00 00 00       	mov    $0x0,%edi	68817: R_X86_64_32	.rodata.str1.1+0x29c1
   6881b:	e8 00 00 00 00       	call   68820 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x890>	6881c: R_X86_64_PLT32	getenv-0x4
   68820:	48 85 c0             	test   %rax,%rax
   68823:	0f 84 88 01 00 00    	je     689b1 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa21>
   68829:	48 89 c7             	mov    %rax,%rdi
   6882c:	31 f6                	xor    %esi,%esi
   6882e:	ba 0a 00 00 00       	mov    $0xa,%edx
   68833:	e8 00 00 00 00       	call   68838 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8a8>	68834: R_X86_64_PLT32	__isoc23_strtol-0x4
   68838:	85 c0                	test   %eax,%eax
   6883a:	0f 95 c0             	setne  %al
   6883d:	e9 71 01 00 00       	jmp    689b3 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa23>
   68842:	bf 00 00 00 00       	mov    $0x0,%edi	68843: R_X86_64_32	.bss+0x358
   68847:	e8 00 00 00 00       	call   6884c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8bc>	68848: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   6884c:	85 c0                	test   %eax,%eax
   6884e:	0f 84 a8 f8 ff ff    	je     680fc <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x16c>
   68854:	bf 00 00 00 00       	mov    $0x0,%edi	68855: R_X86_64_32	.rodata.str1.1+0x29d4
   68859:	e8 00 00 00 00       	call   6885e <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8ce>	6885a: R_X86_64_PLT32	getenv-0x4
   6885e:	48 85 c0             	test   %rax,%rax
   68861:	0f 84 83 02 00 00    	je     68aea <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb5a>
   68867:	48 89 c7             	mov    %rax,%rdi
   6886a:	31 f6                	xor    %esi,%esi
   6886c:	ba 0a 00 00 00       	mov    $0xa,%edx
   68871:	e8 00 00 00 00       	call   68876 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8e6>	68872: R_X86_64_PLT32	__isoc23_strtol-0x4
   68876:	e9 74 02 00 00       	jmp    68aef <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb5f>
   6887b:	bf 10 00 00 00       	mov    $0x10,%edi
   68880:	e8 00 00 00 00       	call   68885 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x8f5>	68881: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68885:	49 89 c6             	mov    %rax,%r14
   68888:	be 00 00 00 00       	mov    $0x0,%esi	68889: R_X86_64_32	.rodata.str1.1+0x45e
   6888d:	48 89 c7             	mov    %rax,%rdi
   68890:	e8 00 00 00 00       	call   68895 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x905>	68891: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68895:	e9 9b 02 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   6889a:	bf 00 00 00 00       	mov    $0x0,%edi	6889b: R_X86_64_32	.bss+0x348
   6889f:	e8 00 00 00 00       	call   688a4 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x914>	688a0: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   688a4:	85 c0                	test   %eax,%eax
   688a6:	0f 84 fc fa ff ff    	je     683a8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x418>
   688ac:	bf 00 00 00 00       	mov    $0x0,%edi	688ad: R_X86_64_32	.rodata.str1.1+0x29c1
   688b1:	e8 00 00 00 00       	call   688b6 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x926>	688b2: R_X86_64_PLT32	getenv-0x4
   688b6:	48 85 c0             	test   %rax,%rax
   688b9:	0f 84 45 02 00 00    	je     68b04 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb74>
   688bf:	48 89 c7             	mov    %rax,%rdi
   688c2:	31 f6                	xor    %esi,%esi
   688c4:	ba 0a 00 00 00       	mov    $0xa,%edx
   688c9:	e8 00 00 00 00       	call   688ce <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x93e>	688ca: R_X86_64_PLT32	__isoc23_strtol-0x4
   688ce:	85 c0                	test   %eax,%eax
   688d0:	0f 95 c0             	setne  %al
   688d3:	e9 2e 02 00 00       	jmp    68b06 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb76>
   688d8:	bf 10 00 00 00       	mov    $0x10,%edi
   688dd:	e8 00 00 00 00       	call   688e2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x952>	688de: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   688e2:	49 89 c6             	mov    %rax,%r14
   688e5:	be 00 00 00 00       	mov    $0x0,%esi	688e6: R_X86_64_32	.rodata.str1.1+0x2cb
   688ea:	48 89 c7             	mov    %rax,%rdi
   688ed:	e8 00 00 00 00       	call   688f2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x962>	688ee: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   688f2:	e9 3e 02 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   688f7:	bf 10 00 00 00       	mov    $0x10,%edi
   688fc:	e8 00 00 00 00       	call   68901 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x971>	688fd: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68901:	49 89 c6             	mov    %rax,%r14
   68904:	be 00 00 00 00       	mov    $0x0,%esi	68905: R_X86_64_32	.rodata.str1.1+0x2cb
   68909:	48 89 c7             	mov    %rax,%rdi
   6890c:	e8 00 00 00 00       	call   68911 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x981>	6890d: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68911:	e9 1f 02 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68916:	bf 10 00 00 00       	mov    $0x10,%edi
   6891b:	e8 00 00 00 00       	call   68920 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x990>	6891c: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68920:	49 89 c6             	mov    %rax,%r14
   68923:	be 00 00 00 00       	mov    $0x0,%esi	68924: R_X86_64_32	.rodata.str1.1+0x2cb
   68928:	48 89 c7             	mov    %rax,%rdi
   6892b:	e8 00 00 00 00       	call   68930 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9a0>	6892c: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68930:	e9 00 02 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68935:	bf 00 00 00 00       	mov    $0x0,%edi	68936: R_X86_64_32	.bss+0x358
   6893a:	e8 00 00 00 00       	call   6893f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9af>	6893b: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   6893f:	85 c0                	test   %eax,%eax
   68941:	0f 84 7d fa ff ff    	je     683c4 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x434>
   68947:	bf 00 00 00 00       	mov    $0x0,%edi	68948: R_X86_64_32	.rodata.str1.1+0x29d4
   6894c:	e8 00 00 00 00       	call   68951 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9c1>	6894d: R_X86_64_PLT32	getenv-0x4
   68951:	48 85 c0             	test   %rax,%rax
   68954:	0f 84 67 02 00 00    	je     68bc1 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc31>
   6895a:	48 89 c7             	mov    %rax,%rdi
   6895d:	31 f6                	xor    %esi,%esi
   6895f:	ba 0a 00 00 00       	mov    $0xa,%edx
   68964:	e8 00 00 00 00       	call   68969 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9d9>	68965: R_X86_64_PLT32	__isoc23_strtol-0x4
   68969:	e9 58 02 00 00       	jmp    68bc6 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc36>
   6896e:	bf 00 00 00 00       	mov    $0x0,%edi	6896f: R_X86_64_32	.bss+0x368
   68973:	e8 00 00 00 00       	call   68978 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9e8>	68974: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68978:	85 c0                	test   %eax,%eax
   6897a:	0f 84 98 f7 ff ff    	je     68118 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x188>
   68980:	e8 db 9c 00 00       	call   72660 <_ZZN6strata7kernels12_GLOBAL__N_125q5_head_runtime_ones_wordEvENK4wordMUlvE_clEv>
   68985:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 6898b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x9fb>	68987: R_X86_64_PC32	.bss+0x35c
   6898b:	bf 00 00 00 00       	mov    $0x0,%edi	6898c: R_X86_64_32	.bss+0x368
   68990:	e8 00 00 00 00       	call   68995 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa05>	68991: R_X86_64_PLT32	__cxa_guard_release-0x4
   68995:	e9 7e f7 ff ff       	jmp    68118 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x188>
   6899a:	b0 01                	mov    $0x1,%al
   6899c:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 689a2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa12>	6899e: R_X86_64_PC32	.bss+0x33c
   689a2:	bf 00 00 00 00       	mov    $0x0,%edi	689a3: R_X86_64_32	.bss+0x348
   689a7:	e8 00 00 00 00       	call   689ac <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa1c>	689a8: R_X86_64_PLT32	__cxa_guard_release-0x4
   689ac:	e9 2f f7 ff ff       	jmp    680e0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x150>
   689b1:	b0 01                	mov    $0x1,%al
   689b3:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 689b9 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa29>	689b5: R_X86_64_PC32	.bss+0x33c
   689b9:	bf 00 00 00 00       	mov    $0x0,%edi	689ba: R_X86_64_32	.bss+0x348
   689be:	e8 00 00 00 00       	call   689c3 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa33>	689bf: R_X86_64_PLT32	__cxa_guard_release-0x4
   689c3:	e9 5d f6 ff ff       	jmp    68025 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x95>
   689c8:	bf 10 00 00 00       	mov    $0x10,%edi
   689cd:	e8 00 00 00 00       	call   689d2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa42>	689ce: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   689d2:	49 89 c6             	mov    %rax,%r14
   689d5:	be 00 00 00 00       	mov    $0x0,%esi	689d6: R_X86_64_32	.rodata.str1.1+0x2cb
   689da:	48 89 c7             	mov    %rax,%rdi
   689dd:	e8 00 00 00 00       	call   689e2 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa52>	689de: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   689e2:	e9 4e 01 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   689e7:	bf 10 00 00 00       	mov    $0x10,%edi
   689ec:	e8 00 00 00 00       	call   689f1 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa61>	689ed: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   689f1:	49 89 c6             	mov    %rax,%r14
   689f4:	be 00 00 00 00       	mov    $0x0,%esi	689f5: R_X86_64_32	.rodata.str1.1+0x2cb
   689f9:	48 89 c7             	mov    %rax,%rdi
   689fc:	e8 00 00 00 00       	call   68a01 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa71>	689fd: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68a01:	e9 2f 01 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68a06:	bf 10 00 00 00       	mov    $0x10,%edi
   68a0b:	e8 00 00 00 00       	call   68a10 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa80>	68a0c: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68a10:	49 89 c6             	mov    %rax,%r14
   68a13:	be 00 00 00 00       	mov    $0x0,%esi	68a14: R_X86_64_32	.rodata.str1.1+0x2cb
   68a18:	48 89 c7             	mov    %rax,%rdi
   68a1b:	e8 00 00 00 00       	call   68a20 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa90>	68a1c: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68a20:	e9 10 01 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68a25:	bf 10 00 00 00       	mov    $0x10,%edi
   68a2a:	e8 00 00 00 00       	call   68a2f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xa9f>	68a2b: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68a2f:	49 89 c6             	mov    %rax,%r14
   68a32:	be 00 00 00 00       	mov    $0x0,%esi	68a33: R_X86_64_32	.rodata.str1.1+0x45e
   68a37:	48 89 c7             	mov    %rax,%rdi
   68a3a:	e8 00 00 00 00       	call   68a3f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xaaf>	68a3b: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68a3f:	e9 f1 00 00 00       	jmp    68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>
   68a44:	bf 00 00 00 00       	mov    $0x0,%edi	68a45: R_X86_64_32	.bss+0x68
   68a49:	e8 00 00 00 00       	call   68a4e <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xabe>	68a4a: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68a4e:	85 c0                	test   %eax,%eax
   68a50:	0f 84 2c f7 ff ff    	je     68182 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x1f2>
   68a56:	48 8b 0d 00 00 00 00 	mov    0x0(%rip),%rcx        # 68a5d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xacd>	68a59: R_X86_64_PC32	stderr-0x4
   68a5d:	bf 00 00 00 00       	mov    $0x0,%edi	68a5e: R_X86_64_32	.rodata.str1.1+0x2a42
   68a62:	be 63 00 00 00       	mov    $0x63,%esi
   68a67:	ba 01 00 00 00       	mov    $0x1,%edx
   68a6c:	e8 00 00 00 00       	call   68a71 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xae1>	68a6d: R_X86_64_PLT32	fwrite-0x4
   68a71:	c6 05 00 00 00 00 01 	movb   $0x1,0x0(%rip)        # 68a78 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xae8>	68a73: R_X86_64_PC32	.bss+0x5b
   68a78:	bf 00 00 00 00       	mov    $0x0,%edi	68a79: R_X86_64_32	.bss+0x68
   68a7d:	e8 00 00 00 00       	call   68a82 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xaf2>	68a7e: R_X86_64_PLT32	__cxa_guard_release-0x4
   68a82:	e9 fb f6 ff ff       	jmp    68182 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x1f2>
   68a87:	bf 00 00 00 00       	mov    $0x0,%edi	68a88: R_X86_64_32	.bss+0x378
   68a8c:	e8 00 00 00 00       	call   68a91 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb01>	68a8d: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68a91:	85 c0                	test   %eax,%eax
   68a93:	0f 84 0c f8 ff ff    	je     682a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x315>
   68a99:	bf 00 00 00 00       	mov    $0x0,%edi	68a9a: R_X86_64_32	.rodata+0x10e08
   68a9e:	e8 00 00 00 00       	call   68aa3 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb13>	68a9f: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   68aa3:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 68aaa <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb1a>	68aa6: R_X86_64_PC32	.bss+0x36c
   68aaa:	bf 00 00 00 00       	mov    $0x0,%edi	68aab: R_X86_64_32	.bss+0x378
   68aaf:	e8 00 00 00 00       	call   68ab4 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb24>	68ab0: R_X86_64_PLT32	__cxa_guard_release-0x4
   68ab4:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
   68ab9:	e9 e7 f7 ff ff       	jmp    682a5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x315>
   68abe:	bf 00 00 00 00       	mov    $0x0,%edi	68abf: R_X86_64_32	.bss+0x388
   68ac3:	e8 00 00 00 00       	call   68ac8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb38>	68ac4: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68ac8:	85 c0                	test   %eax,%eax
   68aca:	0f 84 10 f9 ff ff    	je     683e0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x450>
   68ad0:	e8 eb 9c 00 00       	call   727c0 <_ZZN6strata7kernels12_GLOBAL__N_128q5_head_runtime_ones_c6_wordEvENK4wordMUlvE_clEv>
   68ad5:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 68adb <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb4b>	68ad7: R_X86_64_PC32	.bss+0x37c
   68adb:	bf 00 00 00 00       	mov    $0x0,%edi	68adc: R_X86_64_32	.bss+0x388
   68ae0:	e8 00 00 00 00       	call   68ae5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb55>	68ae1: R_X86_64_PLT32	__cxa_guard_release-0x4
   68ae5:	e9 f6 f8 ff ff       	jmp    683e0 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x450>
   68aea:	b8 01 00 00 00       	mov    $0x1,%eax
   68aef:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 68af5 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb65>	68af1: R_X86_64_PC32	.bss+0x34c
   68af5:	bf 00 00 00 00       	mov    $0x0,%edi	68af6: R_X86_64_32	.bss+0x358
   68afa:	e8 00 00 00 00       	call   68aff <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb6f>	68afb: R_X86_64_PLT32	__cxa_guard_release-0x4
   68aff:	e9 f8 f5 ff ff       	jmp    680fc <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x16c>
   68b04:	b0 01                	mov    $0x1,%al
   68b06:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 68b0c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb7c>	68b08: R_X86_64_PC32	.bss+0x33c
   68b0c:	bf 00 00 00 00       	mov    $0x0,%edi	68b0d: R_X86_64_32	.bss+0x348
   68b11:	e8 00 00 00 00       	call   68b16 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb86>	68b12: R_X86_64_PLT32	__cxa_guard_release-0x4
   68b16:	e9 8d f8 ff ff       	jmp    683a8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x418>
   68b1b:	bf 10 00 00 00       	mov    $0x10,%edi
   68b20:	e8 00 00 00 00       	call   68b25 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xb95>	68b21: R_X86_64_PLT32	__cxa_allocate_exception-0x4
   68b25:	49 89 c6             	mov    %rax,%r14
   68b28:	be 00 00 00 00       	mov    $0x0,%esi	68b29: R_X86_64_32	.rodata.str1.1+0x45e
   68b2d:	48 89 c7             	mov    %rax,%rdi
   68b30:	e8 00 00 00 00       	call   68b35 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xba5>	68b31: R_X86_64_PLT32	_ZNSt16invalid_argumentC1EPKc-0x4
   68b35:	be 00 00 00 00       	mov    $0x0,%esi	68b36: R_X86_64_32	_ZTISt16invalid_argument
   68b3a:	ba 00 00 00 00       	mov    $0x0,%edx	68b3b: R_X86_64_32	_ZNSt16invalid_argumentD1Ev
   68b3f:	4c 89 f7             	mov    %r14,%rdi
   68b42:	e8 00 00 00 00       	call   68b47 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbb7>	68b43: R_X86_64_PLT32	__cxa_throw-0x4
   68b47:	bf 00 00 00 00       	mov    $0x0,%edi	68b48: R_X86_64_32	.bss+0x78
   68b4c:	e8 00 00 00 00       	call   68b51 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbc1>	68b4d: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68b51:	85 c0                	test   %eax,%eax
   68b53:	0f 84 f1 f8 ff ff    	je     6844a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x4ba>
   68b59:	48 8b 0d 00 00 00 00 	mov    0x0(%rip),%rcx        # 68b60 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbd0>	68b5c: R_X86_64_PC32	stderr-0x4
   68b60:	bf 00 00 00 00       	mov    $0x0,%edi	68b61: R_X86_64_32	.rodata.str1.1+0x2b9a
   68b65:	be 66 00 00 00       	mov    $0x66,%esi
   68b6a:	ba 01 00 00 00       	mov    $0x1,%edx
   68b6f:	e8 00 00 00 00       	call   68b74 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbe4>	68b70: R_X86_64_PLT32	fwrite-0x4
   68b74:	c6 05 00 00 00 00 01 	movb   $0x1,0x0(%rip)        # 68b7b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbeb>	68b76: R_X86_64_PC32	.bss+0x6b
   68b7b:	bf 00 00 00 00       	mov    $0x0,%edi	68b7c: R_X86_64_32	.bss+0x78
   68b80:	e8 00 00 00 00       	call   68b85 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xbf5>	68b81: R_X86_64_PLT32	__cxa_guard_release-0x4
   68b85:	e9 c0 f8 ff ff       	jmp    6844a <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x4ba>
   68b8a:	bf 00 00 00 00       	mov    $0x0,%edi	68b8b: R_X86_64_32	.bss+0x398
   68b8f:	e8 00 00 00 00       	call   68b94 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc04>	68b90: R_X86_64_PLT32	__cxa_guard_acquire-0x4
   68b94:	85 c0                	test   %eax,%eax
   68b96:	0f 84 d1 f9 ff ff    	je     6856d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x5dd>
   68b9c:	bf 00 00 00 00       	mov    $0x0,%edi	68b9d: R_X86_64_32	.rodata+0x11020
   68ba1:	e8 00 00 00 00       	call   68ba6 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc16>	68ba2: R_X86_64_PLT32	_ZN4sycl3_V16detail19getDeviceKernelInfoERKNS1_27compile_time_kernel_info_v123CompileTimeKernelInfoTyE-0x4
   68ba6:	48 89 05 00 00 00 00 	mov    %rax,0x0(%rip)        # 68bad <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc1d>	68ba9: R_X86_64_PC32	.bss+0x38c
   68bad:	bf 00 00 00 00       	mov    $0x0,%edi	68bae: R_X86_64_32	.bss+0x398
   68bb2:	e8 00 00 00 00       	call   68bb7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc27>	68bb3: R_X86_64_PLT32	__cxa_guard_release-0x4
   68bb7:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
   68bbc:	e9 ac f9 ff ff       	jmp    6856d <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x5dd>
   68bc1:	b8 01 00 00 00       	mov    $0x1,%eax
   68bc6:	89 05 00 00 00 00    	mov    %eax,0x0(%rip)        # 68bcc <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc3c>	68bc8: R_X86_64_PC32	.bss+0x34c
   68bcc:	bf 00 00 00 00       	mov    $0x0,%edi	68bcd: R_X86_64_32	.bss+0x358
   68bd1:	e8 00 00 00 00       	call   68bd6 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc46>	68bd2: R_X86_64_PLT32	__cxa_guard_release-0x4
   68bd6:	e9 e9 f7 ff ff       	jmp    683c4 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0x434>
   68bdb:	48 89 c3             	mov    %rax,%rbx
   68bde:	bf 00 00 00 00       	mov    $0x0,%edi	68bdf: R_X86_64_32	.bss+0x398
   68be3:	e8 00 00 00 00       	call   68be8 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc58>	68be4: R_X86_64_PLT32	__cxa_guard_abort-0x4
   68be8:	eb 52                	jmp    68c3c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcac>
   68bea:	48 89 c3             	mov    %rax,%rbx
   68bed:	bf 00 00 00 00       	mov    $0x0,%edi	68bee: R_X86_64_32	.bss+0x388
   68bf2:	e8 00 00 00 00       	call   68bf7 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc67>	68bf3: R_X86_64_PLT32	__cxa_guard_abort-0x4
   68bf7:	48 89 df             	mov    %rbx,%rdi
   68bfa:	e8 00 00 00 00       	call   68bff <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc6f>	68bfb: R_X86_64_PLT32	_Unwind_Resume-0x4
   68bff:	48 89 c3             	mov    %rax,%rbx
   68c02:	bf 00 00 00 00       	mov    $0x0,%edi	68c03: R_X86_64_32	.bss+0x378
   68c07:	e8 00 00 00 00       	call   68c0c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc7c>	68c08: R_X86_64_PLT32	__cxa_guard_abort-0x4
   68c0c:	eb 2e                	jmp    68c3c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcac>
   68c0e:	48 89 c3             	mov    %rax,%rbx
   68c11:	bf 00 00 00 00       	mov    $0x0,%edi	68c12: R_X86_64_32	.bss+0x368
   68c16:	e8 00 00 00 00       	call   68c1b <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc8b>	68c17: R_X86_64_PLT32	__cxa_guard_abort-0x4
   68c1b:	48 89 df             	mov    %rbx,%rdi
   68c1e:	e8 00 00 00 00       	call   68c23 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xc93>	68c1f: R_X86_64_PLT32	_Unwind_Resume-0x4
   68c23:	eb 37                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c25:	eb 35                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c27:	eb 33                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c29:	eb 31                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c2b:	eb 2f                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c2d:	eb 0a                	jmp    68c39 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xca9>
   68c2f:	eb 2b                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c31:	eb 29                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c33:	eb 27                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c35:	eb 25                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c37:	eb 36                	jmp    68c6f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcdf>
   68c39:	48 89 c3             	mov    %rax,%rbx
   68c3c:	48 8d 7c 24 30       	lea    0x30(%rsp),%rdi
   68c41:	e8 00 00 00 00       	call   68c46 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcb6>	68c42: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   68c46:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
   68c4b:	e8 00 00 00 00       	call   68c50 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcc0>	68c4c: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   68c50:	48 89 df             	mov    %rbx,%rdi
   68c53:	e8 00 00 00 00       	call   68c58 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcc8>	68c54: R_X86_64_PLT32	_Unwind_Resume-0x4
   68c58:	eb 02                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c5a:	eb 00                	jmp    68c5c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xccc>
   68c5c:	48 89 c3             	mov    %rax,%rbx
   68c5f:	4c 89 f7             	mov    %r14,%rdi
   68c62:	e8 00 00 00 00       	call   68c67 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcd7>	68c63: R_X86_64_PLT32	__cxa_free_exception-0x4
   68c67:	48 89 df             	mov    %rbx,%rdi
   68c6a:	e8 00 00 00 00       	call   68c6f <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcdf>	68c6b: R_X86_64_PLT32	_Unwind_Resume-0x4
   68c6f:	48 89 c3             	mov    %rax,%rbx
   68c72:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
   68c77:	e8 00 00 00 00       	call   68c7c <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcec>	68c78: R_X86_64_PLT32	_ZN4sycl3_V16detail14tls_code_loc_tD1Ev-0x4
   68c7c:	48 89 df             	mov    %rbx,%rdi
   68c7f:	e8 00 00 00 00       	call   68c84 <_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv+0xcf4>	68c80: R_X86_64_PLT32	_Unwind_Resume-0x4

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
