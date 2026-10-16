
/mnt/vm_8tb/b70/build/strata-ab/src-ep/build-sycl-ep/CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o:     file format elf64-x86-64


Disassembly of section .text:

000000000000dd20 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
    dd20:	55                   	push   %rbp
    dd21:	41 57                	push   %r15
    dd23:	41 56                	push   %r14
    dd25:	41 55                	push   %r13
    dd27:	41 54                	push   %r12
    dd29:	53                   	push   %rbx
    dd2a:	48 83 ec 78          	sub    $0x78,%rsp
    dd2e:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    dd33:	49 89 d6             	mov    %rdx,%r14
    dd36:	49 89 f4             	mov    %rsi,%r12
    dd39:	49 89 ff             	mov    %rdi,%r15
    dd3c:	48 8b 9f f0 05 00 00 	mov    0x5f0(%rdi),%rbx
    dd43:	48 8d 87 00 06 00 00 	lea    0x600(%rdi),%rax
    dd4a:	48 8d 8f c0 02 00 00 	lea    0x2c0(%rdi),%rcx
    dd51:	48 85 db             	test   %rbx,%rbx
    dd54:	48 0f 44 c1          	cmove  %rcx,%rax
    dd58:	4c 8b 28             	mov    (%rax),%r13
    dd5b:	e8 00 00 00 00       	call   dd60 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x40>	dd5c: R_X86_64_PLT32	_ZN6strata7kernels18argmax_rows_wantedEv-0x4
    dd60:	84 c0                	test   %al,%al
    dd62:	74 26                	je     dd8a <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x6a>
    dd64:	49 8b 8f 40 06 00 00 	mov    0x640(%r15),%rcx
    dd6b:	4d 8b 87 78 07 00 00 	mov    0x778(%r15),%r8
    dd72:	4c 89 e7             	mov    %r12,%rdi
    dd75:	be 01 00 00 00       	mov    $0x1,%esi
    dd7a:	44 89 ea             	mov    %r13d,%edx
    dd7d:	4d 89 f1             	mov    %r14,%r9
    dd80:	e8 00 00 00 00       	call   dd85 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x65>	dd81: R_X86_64_PLT32	_ZN6strata7kernels11argmax_rowsEPKfiiPvPiS3_-0x4
    dd85:	e9 cf 00 00 00       	jmp    de59 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x139>
    dd8a:	c7 44 24 38 14 00 00 00 	movl   $0x14,0x38(%rsp)
    dd92:	f3 0f 10 05 00 00 00 00 	movss  0x0(%rip),%xmm0        # dd9a <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x7a>	dd96: R_X86_64_PC32	.LCPI73_0-0x4
    dd9a:	0f 13 44 24 3c       	movlps %xmm0,0x3c(%rsp)
    dd9f:	c7 44 24 4c 00 00 00 00 	movl   $0x0,0x4c(%rsp)
    dda7:	f3 0f 10 05 00 00 00 00 	movss  0x0(%rip),%xmm0        # ddaf <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x8f>	ddab: R_X86_64_PC32	.LCPI73_1-0x4
    ddaf:	0f 13 44 24 50       	movlps %xmm0,0x50(%rsp)
    ddb4:	c7 44 24 58 00 00 00 00 	movl   $0x0,0x58(%rsp)
    ddbc:	0f 57 c0             	xorps  %xmm0,%xmm0
    ddbf:	0f 11 44 24 60       	movups %xmm0,0x60(%rsp)
    ddc4:	c6 44 24 70 01       	movb   $0x1,0x70(%rsp)
    ddc9:	48 b8 00 00 00 00 01 00 00 00 	movabs $0x100000000,%rax
    ddd3:	48 89 44 24 44       	mov    %rax,0x44(%rsp)
    ddd8:	48 85 db             	test   %rbx,%rbx
    dddb:	75 55                	jne    de32 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x112>
    dddd:	49 81 fd 00 ca 03 00 	cmp    $0x3ca00,%r13
    dde4:	75 4c                	jne    de32 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x112>
    dde6:	41 80 bf f1 00 00 00 00 	cmpb   $0x0,0xf1(%r15)
    ddee:	75 42                	jne    de32 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x112>
    ddf0:	41 80 bf f2 00 00 00 00 	cmpb   $0x0,0xf2(%r15)
    ddf8:	75 38                	jne    de32 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x112>
    ddfa:	e8 f1 ba ff ff       	call   98f0 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv>
    ddff:	84 c0                	test   %al,%al
    de01:	74 2f                	je     de32 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x112>
    de03:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # de0a <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xea>	de06: R_X86_64_PC32	.bss+0x74
    de0a:	84 c0                	test   %al,%al
    de0c:	0f 84 c8 00 00 00    	je     deda <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1ba>
    de12:	49 8b 97 40 06 00 00 	mov    0x640(%r15),%rdx
    de19:	49 8b 8f 78 07 00 00 	mov    0x778(%r15),%rcx
    de20:	4c 89 e7             	mov    %r12,%rdi
    de23:	be 00 ca 03 00       	mov    $0x3ca00,%esi
    de28:	4d 89 f0             	mov    %r14,%r8
    de2b:	e8 00 00 00 00       	call   de30 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x110>	de2c: R_X86_64_PLT32	_ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE-0x4
    de30:	eb 27                	jmp    de59 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x139>
    de32:	4c 8d 4c 24 38       	lea    0x38(%rsp),%r9
    de37:	4c 89 e7             	mov    %r12,%rdi
    de3a:	be 01 00 00 00       	mov    $0x1,%esi
    de3f:	44 89 ea             	mov    %r13d,%edx
    de42:	31 c9                	xor    %ecx,%ecx
    de44:	45 31 c0             	xor    %r8d,%r8d
    de47:	41 56                	push   %r14
    de49:	41 ff b7 78 07 00 00 	push   0x778(%r15)
    de50:	e8 00 00 00 00       	call   de55 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x135>	de51: R_X86_64_PLT32	_ZN6strata7kernels13sample_tokensEPKfiiPKiiRKNS0_13SamplerParamsEPiPv-0x4
    de55:	48 83 c4 10          	add    $0x10,%rsp
    de59:	e8 00 00 00 00       	call   de5e <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x13e>	de5a: R_X86_64_PLT32	_ZN6strata7kernels20multi_block_head_opsEv-0x4
    de5e:	4d 8b 87 38 06 00 00 	mov    0x638(%r15),%r8
    de65:	49 8b 8f 78 07 00 00 	mov    0x778(%r15),%rcx
    de6c:	84 c0                	test   %al,%al
    de6e:	74 23                	je     de93 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x173>
    de70:	4d 8b 8f 48 06 00 00 	mov    0x648(%r15),%r9
    de77:	48 83 ec 08          	sub    $0x8,%rsp
    de7b:	4c 89 e7             	mov    %r12,%rdi
    de7e:	be 01 00 00 00       	mov    $0x1,%esi
    de83:	44 89 ea             	mov    %r13d,%edx
    de86:	41 56                	push   %r14
    de88:	e8 00 00 00 00       	call   de8d <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x16d>	de89: R_X86_64_PLT32	_ZN6strata7kernels18row_top_prob_splitEPKfiiPKiPfPvS6_-0x4
    de8d:	48 83 c4 10          	add    $0x10,%rsp
    de91:	eb 13                	jmp    dea6 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x186>
    de93:	4c 89 e7             	mov    %r12,%rdi
    de96:	be 01 00 00 00       	mov    $0x1,%esi
    de9b:	44 89 ea             	mov    %r13d,%edx
    de9e:	4d 89 f1             	mov    %r14,%r9
    dea1:	e8 00 00 00 00       	call   dea6 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x186>	dea2: R_X86_64_PLT32	_ZN6strata7kernels12row_top_probEPKfiiPKiPfPv-0x4
    dea6:	40 b5 01             	mov    $0x1,%bpl
    dea9:	48 85 db             	test   %rbx,%rbx
    deac:	74 1b                	je     dec9 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1a9>
    deae:	49 8b b7 f8 05 00 00 	mov    0x5f8(%r15),%rsi
    deb5:	49 8b bf 78 07 00 00 	mov    0x778(%r15),%rdi
    debc:	ba 01 00 00 00       	mov    $0x1,%edx
    dec1:	4c 89 f1             	mov    %r14,%rcx
    dec4:	e8 00 00 00 00       	call   dec9 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1a9>	dec5: R_X86_64_PLT32	_ZN6strata7kernels7map_idsEPiPKiiPv-0x4
    dec9:	89 e8                	mov    %ebp,%eax
    decb:	48 83 c4 78          	add    $0x78,%rsp
    decf:	5b                   	pop    %rbx
    ded0:	41 5c                	pop    %r12
    ded2:	41 5d                	pop    %r13
    ded4:	41 5e                	pop    %r14
    ded6:	41 5f                	pop    %r15
    ded8:	5d                   	pop    %rbp
    ded9:	c3                   	ret
    deda:	bf 00 00 00 00       	mov    $0x0,%edi	dedb: R_X86_64_32	.bss+0x78
    dedf:	e8 00 00 00 00       	call   dee4 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1c4>	dee0: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    dee4:	85 c0                	test   %eax,%eax
    dee6:	0f 84 26 ff ff ff    	je     de12 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xf2>
    deec:	48 8b 0d 00 00 00 00 	mov    0x0(%rip),%rcx        # def3 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1d3>	deef: R_X86_64_PC32	stderr-0x4
    def3:	bf 00 00 00 00       	mov    $0x0,%edi	def4: R_X86_64_32	.rodata.str1.1+0x20be
    def8:	be 4e 00 00 00       	mov    $0x4e,%esi
    defd:	ba 01 00 00 00       	mov    $0x1,%edx
    df02:	e8 00 00 00 00       	call   df07 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1e7>	df03: R_X86_64_PLT32	fwrite-0x4
    df07:	c6 05 00 00 00 00 01 	movb   $0x1,0x0(%rip)        # df0e <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1ee>	df09: R_X86_64_PC32	.bss+0x6b
    df0e:	bf 00 00 00 00       	mov    $0x0,%edi	df0f: R_X86_64_32	.bss+0x78
    df13:	e8 00 00 00 00       	call   df18 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1f8>	df14: R_X86_64_PLT32	__cxa_guard_release-0x4
    df18:	e9 f5 fe ff ff       	jmp    de12 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xf2>
    df1d:	eb 00                	jmp    df1f <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1ff>
    df1f:	49 89 c6             	mov    %rax,%r14
    df22:	83 fa 01             	cmp    $0x1,%edx
    df25:	0f 85 b6 00 00 00    	jne    dfe1 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c1>
    df2b:	4c 89 f7             	mov    %r14,%rdi
    df2e:	e8 00 00 00 00       	call   df33 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x213>	df2f: R_X86_64_PLT32	__cxa_begin_catch-0x4
    df33:	49 89 c6             	mov    %rax,%r14
    df36:	48 8d 7c 24 18       	lea    0x18(%rsp),%rdi
    df3b:	48 8d 54 24 0f       	lea    0xf(%rsp),%rdx
    df40:	be 00 00 00 00       	mov    $0x0,%esi	df41: R_X86_64_32	.rodata.str1.1+0x3d9
    df45:	e8 00 00 00 00       	call   df4a <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x22a>	df46: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_-0x4
    df4a:	49 8b 06             	mov    (%r14),%rax
    df4d:	4c 89 f7             	mov    %r14,%rdi
    df50:	ff 50 10             	call   *0x10(%rax)
    df53:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    df58:	48 8d 74 24 18       	lea    0x18(%rsp),%rsi
    df5d:	48 89 c2             	mov    %rax,%rdx
    df60:	e8 00 00 00 00       	call   df65 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x245>	df61: R_X86_64_PLT32	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_-0x4
    df65:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
    df6a:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    df6f:	e8 00 00 00 00       	call   df74 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x254>	df70: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_-0x4
    df74:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    df79:	48 8d 44 24 48       	lea    0x48(%rsp),%rax
    df7e:	48 39 c7             	cmp    %rax,%rdi
    df81:	74 0d                	je     df90 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x270>
    df83:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    df88:	48 ff c6             	inc    %rsi
    df8b:	e8 00 00 00 00       	call   df90 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x270>	df8c: R_X86_64_PLT32	_ZdlPvm-0x4
    df90:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    df95:	48 8d 44 24 28       	lea    0x28(%rsp),%rax
    df9a:	48 39 c7             	cmp    %rax,%rdi
    df9d:	74 0d                	je     dfac <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x28c>
    df9f:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    dfa4:	48 ff c6             	inc    %rsi
    dfa7:	e8 00 00 00 00       	call   dfac <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x28c>	dfa8: R_X86_64_PLT32	_ZdlPvm-0x4
    dfac:	e8 00 00 00 00       	call   dfb1 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x291>	dfad: R_X86_64_PLT32	__cxa_end_catch-0x4
    dfb1:	31 ed                	xor    %ebp,%ebp
    dfb3:	e9 11 ff ff ff       	jmp    dec9 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1a9>
    dfb8:	49 89 c6             	mov    %rax,%r14
    dfbb:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    dfc0:	48 8d 44 24 28       	lea    0x28(%rsp),%rax
    dfc5:	48 39 c7             	cmp    %rax,%rdi
    dfc8:	74 12                	je     dfdc <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bc>
    dfca:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    dfcf:	48 ff c6             	inc    %rsi
    dfd2:	e8 00 00 00 00       	call   dfd7 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2b7>	dfd3: R_X86_64_PLT32	_ZdlPvm-0x4
    dfd7:	eb 03                	jmp    dfdc <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bc>
    dfd9:	49 89 c6             	mov    %rax,%r14
    dfdc:	e8 00 00 00 00       	call   dfe1 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c1>	dfdd: R_X86_64_PLT32	__cxa_end_catch-0x4
    dfe1:	4c 89 f7             	mov    %r14,%rdi
    dfe4:	e8 00 00 00 00       	call   dfe9 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c9>	dfe5: R_X86_64_PLT32	_Unwind_Resume-0x4
    dfe9:	48 89 c7             	mov    %rax,%rdi
    dfec:	e8 00 00 00 00       	call   dff1 <_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2d1>	dfed: R_X86_64_PLT32	__clang_call_terminate-0x4

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt5stackIjSt5dequeIjSaIjEEED2Ev:

Disassembly of section .text.startup:

Disassembly of section .text._ZNSt3mapIPKvSt8functionIFvN4dpct4dim3ES4_PPvjPN4sycl3_V15queueEEESt4lessIS1_ESaISt4pairIKS1_SC_EEED2Ev:

Disassembly of section .text._ZN4sycl3_V15queue4waitERKNS0_6detail13code_locationE:

Disassembly of section .text.__clang_call_terminate:

Disassembly of section .text._ZN4sycl3_V13ext6oneapi12experimental6detail24executable_command_graphD2Ev:

Disassembly of section .text._ZN4dpct10device_ext13destroy_queueERPN4sycl3_V15queueE:

Disassembly of section .text._ZNSt6vectorIfSaIfEE6resizeEm:

Disassembly of section .text._ZN6strata21malloc_device_guardedEmRN4sycl3_V15queueEPKc:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEPKc:

Disassembly of section .text._ZN4sycl3_V15eventD2Ev:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_:

Disassembly of section .text._ZNSt6vectorIN6strata4core10MtpDrafter6TensorESaIS3_EEaSERKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EERKS8_PKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_RKS8_:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_:

Disassembly of section .text._ZN6strata4core10MtpDrafter6TensorD2Ev:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_S9_:

Disassembly of section .text._ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEPKS5_OS8_:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_:

Disassembly of section .text._ZNSt7__cxx119to_stringEm:

Disassembly of section .text._ZN4dpct10device_ext21queues_wait_and_throwEv:

Disassembly of section .text._ZN6strata13big_fill_zeroERN4sycl3_V15queueEPvmm:

Disassembly of section .text._ZNSt6vectorIiSaIiEEC2EmRKS0_:

Disassembly of section .text._ZNK6strata4core10MtpDrafter10vocab_fileB5cxx11Ev:

Disassembly of section .text._ZN4sycl3_V16detail16PropertyListBaseD2Ev:

Disassembly of section .text._ZN6strata4core8OnDeviceC2Ei:

Disassembly of section .text._ZNSt6vectorIiSaIiEE6resizeEm:

Disassembly of section .text._ZN6strata4core8OnDeviceD2Ev:

Disassembly of section .text._ZN12s112_private6launchEPKfiPvPiRN4sycl3_V15queueE:

Disassembly of section .text._ZNSt7__cxx119to_stringEi:

Disassembly of section .text._ZN4sycl3_V113malloc_deviceIfEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN6strata4core13host_boundary4markEPKcii:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_St8functionIFvN4dpct4dim3ES6_PPvjPN4sycl3_V15queueEEEESt10_Select1stISF_ESt4lessIS1_ESaISF_EE8_M_eraseEPSt13_Rb_tree_nodeISF_E:

Disassembly of section .text._ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE24_M_release_last_use_coldEv:

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

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EED2Ev:

Disassembly of section .text._ZN4sycl3_V17contextD2Ev:

Disassembly of section .text._ZN4dpct10device_ext17create_queue_implIJN4sycl3_V18property5queue8in_orderEEEEPNS3_5queueEbDpT_:

Disassembly of section .text._ZNSt6vectorISt10shared_ptrIN4sycl3_V15queueEESaIS4_EE9push_backEOS4_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE9_M_invokeERKSt9_Any_dataOS2_:

Disassembly of section .text._ZNSt17_Function_handlerIFvN4sycl3_V114exception_listEEN4dpct17exception_handlerMUlS2_E_EE10_M_managerERSt9_Any_dataRKS8_St18_Manager_operation:

Disassembly of section .text._ZN4sycl3_V114exception_listD2Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EED0Ev:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_disposeEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:

Disassembly of section .text._ZNSt23_Sp_counted_ptr_inplaceIN4sycl3_V15queueESaIvELN9__gnu_cxx12_Lock_policyE2EE14_M_get_deleterERKSt9type_info:

Disassembly of section .text._ZNK4dpct7dev_mgr8check_idEj:

Disassembly of section .text._ZTWN4dpct7dev_mgr10_dev_stackE:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V15queueELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZN4sycl3_V110get_nativeILNS0_7backendE2ENS0_7contextEEENS0_14backend_traitsIXT_EE11return_typeIT0_EERKS7_:

Disassembly of section .text._ZNSt13unordered_setIPvSt4hashIS0_ESt8equal_toIS0_ESaIS0_EED2Ev:

Disassembly of section .text._ZN4sycl3_V19exceptionC1ESt10error_codePKc:

Disassembly of section .text._ZNSt12__shared_ptrIN4sycl3_V17contextELN9__gnu_cxx12_Lock_policyE2EED2Ev:

Disassembly of section .text._ZSt8_DestroyIPN6strata4core10MtpDrafter6TensorEEvT_S5_:

Disassembly of section .text._ZN6strata21device_alloc_registerEPKvm:

Disassembly of section .text._ZNSt3mapIPKvmSt4lessIS1_ESaISt4pairIKS1_mEEED2Ev:

Disassembly of section .text._ZNSt8_Rb_treeIPKvSt4pairIKS1_mESt10_Select1stIS4_ESt4lessIS1_ESaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E:

Disassembly of section .text._ZN4sycl3_V16detail15getRoundedRangeILi1EEESt5tupleIJNS0_5rangeIXT_EEEbEES5_RKNS0_6deviceE:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EESE_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_RSA_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EESE_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_RSA_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata17arena_alias_checkERNS0_5queueEPvmPmEUlNS0_2idILi1EEEE0_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm:

Disassembly of section .text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm:

Disassembly of section .text._ZN6strata4core10MtpDrafter6TensorC2ERKS2_:

Disassembly of section .text._ZN4dpct10device_ext15get_memory_infoERmS1_:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EESD_S5_Li1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EES5_Li1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelINS1_18RoundedRangeKernelINS0_4itemILi1ELb1EEELi1EZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EES5_Li1EED0Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_EENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_RS9_NS0_4itemILi1ELb1EEELi1EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN6strata13big_fill_zeroERNS0_5queueEPvmmEUlNS0_2idILi1EEEE_NS0_4itemILi1ELb1EEELi1EED0Ev:

Disassembly of section .text._ZZN6strata4core19coupled_draft_stateEvENKUlvE_clEv:

Disassembly of section .text._ZNSt6vectorIN4sycl3_V15eventESaIS2_EED2Ev:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescIN4dpct6detail21dpct_memcpy_3d_detailEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescINS1_19__pf_kernel_wrapperIN4dpct6detail21dpct_memcpy_3d_detailEEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4dpct12sync_barrierEPN4sycl3_V15eventEPNS1_5queueE:

Disassembly of section .text._ZN4sycl3_V16detail18getKernelParamDescI16dpct_kernel_nameIJN12s112_private14GreedyTop1S112EEEEENS1_19kernel_param_desc_tEi:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EED0Ev:

Disassembly of section .text._ZNK4sycl3_V16detail13HostKernelRefIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_RSC_SB_Li3EE19takeOrCopyOwnershipEv:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EE6getPtrEv:

Disassembly of section .text._ZN4sycl3_V16detail14HostKernelBaseD2Ev:

Disassembly of section .text._ZN4sycl3_V16detail10HostKernelIZN12s112_private6launchEPKfiPvPiRNS0_5queueEEUlNS0_7nd_itemILi3EEEE_SB_Li3EED0Ev:

Disassembly of section .text._ZN4dpct12experimental6detail9graph_mgrD2Ev:

Disassembly of section .text._ZN4sycl3_V113malloc_deviceIiEEPT_mRKNS0_5queueERKNS0_13property_listERKNS0_6detail13code_locationE:

Disassembly of section .text._ZN6strata4core13host_boundary5StateD2Ev:
