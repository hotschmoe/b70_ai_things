0000000000016d70 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE>:
   16d70:	55                   	push   %rbp
   16d71:	41 57                	push   %r15
   16d73:	41 56                	push   %r14
   16d75:	41 55                	push   %r13
   16d77:	41 54                	push   %r12
   16d79:	53                   	push   %rbx
   16d7a:	48 83 ec 58          	sub    $0x58,%rsp
   16d7e:	48 63 de             	movslq %esi,%rbx
   16d81:	b0 01                	mov    $0x1,%al
   16d83:	48 83 bc df 68 01 00 00 00 	cmpq   $0x0,0x168(%rdi,%rbx,8)
   16d8c:	0f 85 c8 02 00 00    	jne    1705a <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2ea>
   16d92:	49 89 fe             	mov    %rdi,%r14
   16d95:	48 89 54 24 08       	mov    %rdx,0x8(%rsp)
   16d9a:	48 89 74 24 30       	mov    %rsi,0x30(%rsp)
   16d9f:	4c 8b 2f             	mov    (%rdi),%r13
   16da2:	4c 8b 7f 08          	mov    0x8(%rdi),%r15
   16da6:	49 8b 85 38 01 00 00 	mov    0x138(%r13),%rax
   16dad:	48 8b 28             	mov    (%rax),%rbp
   16db0:	4c 8b a0 98 00 00 00 	mov    0x98(%rax),%r12
   16db7:	4c 0f af e5          	imul   %rbp,%r12
   16dbb:	48 8b 7f 18          	mov    0x18(%rdi),%rdi
   16dbf:	e8 dc 31 00 00       	call   19fa0 <_ZN4dpct12experimentalL15begin_recordingEPN4sycl3_V15queueE>
   16dc4:	49 8b bd 50 07 00 00 	mov    0x750(%r13),%rdi
   16dcb:	4d 8b 46 18          	mov    0x18(%r14),%r8
   16dcf:	49 8b 76 28          	mov    0x28(%r14),%rsi
   16dd3:	89 ea                	mov    %ebp,%edx
   16dd5:	b9 01 00 00 00       	mov    $0x1,%ecx
   16dda:	e8 00 00 00 00       	call   16ddf <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x6f>	16ddb: R_X86_64_PLT32	_ZN6strata7kernels20native_quantize_q8_1EPKfPviiS3_-0x4
   16ddf:	49 8b bf 50 07 00 00 	mov    0x750(%r15),%rdi
   16de6:	4d 8b 46 18          	mov    0x18(%r14),%r8
   16dea:	49 8b 76 28          	mov    0x28(%r14),%rsi
   16dee:	49 03 76 38          	add    0x38(%r14),%rsi
   16df2:	89 ea                	mov    %ebp,%edx
   16df4:	b9 01 00 00 00       	mov    $0x1,%ecx
   16df9:	e8 00 00 00 00       	call   16dfe <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x8e>	16dfa: R_X86_64_PLT32	_ZN6strata7kernels20native_quantize_q8_1EPKfPviiS3_-0x4
   16dfe:	49 8b 85 58 01 00 00 	mov    0x158(%r13),%rax
   16e05:	8b 78 20             	mov    0x20(%rax),%edi
   16e08:	48 8b 30             	mov    (%rax),%rsi
   16e0b:	49 8b 56 28          	mov    0x28(%r14),%rdx
   16e0f:	49 8b 4e 30          	mov    0x30(%r14),%rcx
   16e13:	45 8b 4e 40          	mov    0x40(%r14),%r9d
   16e17:	41 89 e8             	mov    %ebp,%r8d
   16e1a:	41 ff 76 18          	push   0x18(%r14)
   16e1e:	6a 02                	push   $0x2
   16e20:	e8 00 00 00 00       	call   16e25 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xb5>	16e21: R_X86_64_PLT32	_ZN6strata7kernels16native_mmvq_headEiPKvS2_PfiiiPv-0x4
   16e25:	48 83 c4 10          	add    $0x10,%rsp
   16e29:	49 8b 56 18          	mov    0x18(%r14),%rdx
   16e2d:	49 8b 76 30          	mov    0x30(%r14),%rsi
   16e31:	4c 89 ef             	mov    %r13,%rdi
   16e34:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
   16e39:	e8 00 00 00 00       	call   16e3e <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xce>	16e3a: R_X86_64_PLT32	_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE-0x4
   16e3e:	84 c0                	test   %al,%al
   16e40:	0f 84 ea 01 00 00    	je     17030 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c0>
   16e46:	49 8b 56 18          	mov    0x18(%r14),%rdx
   16e4a:	49 8b 76 40          	mov    0x40(%r14),%rsi
   16e4e:	48 c1 e6 02          	shl    $0x2,%rsi
   16e52:	49 03 76 30          	add    0x30(%r14),%rsi
   16e56:	4c 89 ff             	mov    %r15,%rdi
   16e59:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
   16e5e:	e8 00 00 00 00       	call   16e63 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0xf3>	16e5f: R_X86_64_PLT32	_ZN6strata4core10MtpDrafter16record_post_headEPKfPN4sycl3_V15queueERNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE-0x4
   16e63:	84 c0                	test   %al,%al
   16e65:	0f 84 c5 01 00 00    	je     17030 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c0>
   16e6b:	49 8b bd 80 06 00 00 	mov    0x680(%r13),%rdi
   16e72:	49 8b 95 78 07 00 00 	mov    0x778(%r13),%rdx
   16e79:	49 8b 8d 68 06 00 00 	mov    0x668(%r13),%rcx
   16e80:	48 83 c1 04          	add    $0x4,%rcx
   16e84:	4d 8b 85 78 06 00 00 	mov    0x678(%r13),%r8
   16e8b:	4d 8b 8d 50 06 00 00 	mov    0x650(%r13),%r9
   16e92:	48 83 ec 08          	sub    $0x8,%rsp
   16e96:	4c 89 e6             	mov    %r12,%rsi
   16e99:	41 ff b5 e8 05 00 00 	push   0x5e8(%r13)
   16ea0:	41 ff b5 38 06 00 00 	push   0x638(%r13)
   16ea7:	41 ff 76 18          	push   0x18(%r14)
   16eab:	ff 74 24 50          	push   0x50(%rsp)
   16eaf:	41 ff b5 d8 05 00 00 	push   0x5d8(%r13)
   16eb6:	e8 00 00 00 00       	call   16ebb <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x14b>	16eb7: R_X86_64_PLT32	_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_-0x4
   16ebb:	48 83 c4 30          	add    $0x30,%rsp
   16ebf:	49 8b bf 80 06 00 00 	mov    0x680(%r15),%rdi
   16ec6:	49 8b 97 78 07 00 00 	mov    0x778(%r15),%rdx
   16ecd:	49 8b 8f 68 06 00 00 	mov    0x668(%r15),%rcx
   16ed4:	48 83 c1 04          	add    $0x4,%rcx
   16ed8:	4d 8b 87 78 06 00 00 	mov    0x678(%r15),%r8
   16edf:	4d 8b 8f 50 06 00 00 	mov    0x650(%r15),%r9
   16ee6:	40 b5 01             	mov    $0x1,%bpl
   16ee9:	48 83 ec 08          	sub    $0x8,%rsp
   16eed:	4c 89 e6             	mov    %r12,%rsi
   16ef0:	41 ff b7 e8 05 00 00 	push   0x5e8(%r15)
   16ef7:	41 ff b7 38 06 00 00 	push   0x638(%r15)
   16efe:	41 ff 76 18          	push   0x18(%r14)
   16f02:	ff 74 24 50          	push   0x50(%rsp)
   16f06:	41 ff b7 d8 05 00 00 	push   0x5d8(%r15)
   16f0d:	e8 00 00 00 00       	call   16f12 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1a2>	16f0e: R_X86_64_PLT32	_ZN6strata7kernels10mtp_selectEPKflPKiS4_PfPiS6_iPvS2_S5_-0x4
   16f12:	48 83 c4 30          	add    $0x30,%rsp
   16f16:	e9 17 01 00 00       	jmp    17032 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c2>
   16f1b:	49 89 d4             	mov    %rdx,%r12
   16f1e:	49 89 c7             	mov    %rax,%r15
   16f21:	41 83 fc 02          	cmp    $0x2,%r12d
   16f25:	0f 85 77 01 00 00    	jne    170a2 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x332>
   16f2b:	4c 89 ff             	mov    %r15,%rdi
   16f2e:	e8 00 00 00 00       	call   16f33 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1c3>	16f2f: R_X86_64_PLT32	__cxa_begin_catch-0x4
   16f33:	48 8b 08             	mov    (%rax),%rcx
   16f36:	48 89 c7             	mov    %rax,%rdi
   16f39:	ff 51 10             	call   *0x10(%rcx)
   16f3c:	bf 00 00 00 00       	mov    $0x0,%edi	16f3d: R_X86_64_32	_ZSt4cerr
   16f41:	48 89 c6             	mov    %rax,%rsi
   16f44:	e8 00 00 00 00       	call   16f49 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1d9>	16f45: R_X86_64_PLT32	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc-0x4
   16f49:	48 89 c7             	mov    %rax,%rdi
   16f4c:	e8 00 00 00 00       	call   16f51 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1e1>	16f4d: R_X86_64_PLT32	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_-0x4
   16f51:	e8 00 00 00 00       	call   16f56 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x1e6>	16f52: R_X86_64_PLT32	__cxa_end_catch-0x4
   16f56:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
   16f5b:	48 8b 57 08          	mov    0x8(%rdi),%rdx
   16f5f:	b9 00 00 00 00       	mov    $0x0,%ecx	16f60: R_X86_64_32	.rodata.str1.1+0x11d5
   16f64:	41 b8 17 00 00 00    	mov    $0x17,%r8d
   16f6a:	31 f6                	xor    %esi,%esi
   16f6c:	e8 00 00 00 00       	call   16f71 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x201>	16f6d: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm-0x4
   16f71:	31 c0                	xor    %eax,%eax
   16f73:	e9 e2 00 00 00       	jmp    1705a <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2ea>
   16f78:	e9 1f 01 00 00       	jmp    1709c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x32c>
   16f7d:	e9 1a 01 00 00       	jmp    1709c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x32c>
   16f82:	49 89 d4             	mov    %rdx,%r12
   16f85:	49 89 c7             	mov    %rax,%r15
   16f88:	e8 00 00 00 00       	call   16f8d <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x21d>	16f89: R_X86_64_PLT32	__cxa_end_catch-0x4
   16f8d:	e9 10 01 00 00       	jmp    170a2 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x332>
   16f92:	48 89 c7             	mov    %rax,%rdi
   16f95:	e8 00 00 00 00       	call   16f9a <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x22a>	16f96: R_X86_64_PLT32	__clang_call_terminate-0x4
   16f9a:	49 89 d4             	mov    %rdx,%r12
   16f9d:	49 89 c7             	mov    %rax,%r15
   16fa0:	41 83 fc 02          	cmp    $0x2,%r12d
   16fa4:	0f 85 f8 00 00 00    	jne    170a2 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x332>
   16faa:	4c 89 ff             	mov    %r15,%rdi
   16fad:	e8 00 00 00 00       	call   16fb2 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x242>	16fae: R_X86_64_PLT32	__cxa_begin_catch-0x4
   16fb2:	49 89 c7             	mov    %rax,%r15
   16fb5:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
   16fba:	48 8d 54 24 07       	lea    0x7(%rsp),%rdx
   16fbf:	be 00 00 00 00       	mov    $0x0,%esi	16fc0: R_X86_64_32	.rodata.str1.1+0x10a8
   16fc4:	e8 00 00 00 00       	call   16fc9 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x259>	16fc5: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_-0x4
   16fc9:	49 8b 07             	mov    (%r15),%rax
   16fcc:	4c 89 ff             	mov    %r15,%rdi
   16fcf:	ff 50 10             	call   *0x10(%rax)
   16fd2:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
   16fd7:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
   16fdc:	48 89 c2             	mov    %rax,%rdx
   16fdf:	e8 00 00 00 00       	call   16fe4 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x274>	16fe0: R_X86_64_PLT32	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_-0x4
   16fe4:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
   16fe9:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
   16fee:	e8 00 00 00 00       	call   16ff3 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x283>	16fef: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_-0x4
   16ff3:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
   16ff8:	48 8d 44 24 48       	lea    0x48(%rsp),%rax
   16ffd:	48 39 c7             	cmp    %rax,%rdi
   17000:	74 0d                	je     1700f <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x29f>
   17002:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
   17007:	48 ff c6             	inc    %rsi
   1700a:	e8 00 00 00 00       	call   1700f <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x29f>	1700b: R_X86_64_PLT32	_ZdlPvm-0x4
   1700f:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
   17014:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
   17019:	48 39 c7             	cmp    %rax,%rdi
   1701c:	74 0d                	je     1702b <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bb>
   1701e:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
   17023:	48 ff c6             	inc    %rsi
   17026:	e8 00 00 00 00       	call   1702b <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2bb>	17027: R_X86_64_PLT32	_ZdlPvm-0x4
   1702b:	e8 00 00 00 00       	call   17030 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2c0>	1702c: R_X86_64_PLT32	__cxa_end_catch-0x4
   17030:	31 ed                	xor    %ebp,%ebp
   17032:	49 8d 14 de          	lea    (%r14,%rbx,8),%rdx
   17036:	48 81 c2 68 01 00 00 	add    $0x168,%rdx
   1703d:	49 8b 7e 18          	mov    0x18(%r14),%rdi
   17041:	40 0f b6 f5          	movzbl %bpl,%esi
   17045:	b9 00 00 00 00       	mov    $0x0,%ecx	17046: R_X86_64_32	.rodata.str1.1+0x1215
   1704a:	4c 8b 44 24 08       	mov    0x8(%rsp),%r8
   1704f:	41 b9 01 00 00 00    	mov    $0x1,%r9d
   17055:	e8 96 71 ff ff       	call   e1f0 <_ZN6strata4core12_GLOBAL__N_114finish_captureEPN4sycl3_V15queueEbRPNS3_3ext6oneapi12experimental13command_graphILNS8_11graph_stateE1EEEPKcRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb>
   1705a:	48 83 c4 58          	add    $0x58,%rsp
   1705e:	5b                   	pop    %rbx
   1705f:	41 5c                	pop    %r12
   17061:	41 5d                	pop    %r13
   17063:	41 5e                	pop    %r14
   17065:	41 5f                	pop    %r15
   17067:	5d                   	pop    %rbp
   17068:	c3                   	ret
   17069:	eb 31                	jmp    1709c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x32c>
   1706b:	49 89 d4             	mov    %rdx,%r12
   1706e:	49 89 c7             	mov    %rax,%r15
   17071:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
   17076:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
   1707b:	48 39 c7             	cmp    %rax,%rdi
   1707e:	74 15                	je     17095 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x325>
   17080:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
   17085:	48 ff c6             	inc    %rsi
   17088:	e8 00 00 00 00       	call   1708d <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x31d>	17089: R_X86_64_PLT32	_ZdlPvm-0x4
   1708d:	eb 06                	jmp    17095 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x325>
   1708f:	49 89 d4             	mov    %rdx,%r12
   17092:	49 89 c7             	mov    %rax,%r15
   17095:	e8 00 00 00 00       	call   1709a <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x32a>	17096: R_X86_64_PLT32	__cxa_end_catch-0x4
   1709a:	eb 06                	jmp    170a2 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x332>
   1709c:	49 89 d4             	mov    %rdx,%r12
   1709f:	49 89 c7             	mov    %rax,%r15
   170a2:	41 83 fc 01          	cmp    $0x1,%r12d
   170a6:	0f 85 b5 00 00 00    	jne    17161 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f1>
   170ac:	4c 89 ff             	mov    %r15,%rdi
   170af:	e8 00 00 00 00       	call   170b4 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x344>	170b0: R_X86_64_PLT32	__cxa_begin_catch-0x4
   170b4:	49 89 c6             	mov    %rax,%r14
   170b7:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
   170bc:	48 8d 54 24 07       	lea    0x7(%rsp),%rdx
   170c1:	be 00 00 00 00       	mov    $0x0,%esi	170c2: R_X86_64_32	.rodata.str1.1+0x11f8
   170c6:	e8 00 00 00 00       	call   170cb <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x35b>	170c7: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_-0x4
   170cb:	4c 89 f7             	mov    %r14,%rdi
   170ce:	e8 00 00 00 00       	call   170d3 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x363>	170cf: R_X86_64_PLT32	_ZNK4sycl3_V19exception4whatEv-0x4
   170d3:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
   170d8:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
   170dd:	48 89 c2             	mov    %rax,%rdx
   170e0:	e8 00 00 00 00       	call   170e5 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x375>	170e1: R_X86_64_PLT32	_ZStplIcSt11char_traitsIcESaIcEENSt7__cxx1112basic_stringIT_T0_T1_EEOS8_PKS5_-0x4
   170e5:	48 8d 74 24 38       	lea    0x38(%rsp),%rsi
   170ea:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
   170ef:	e8 00 00 00 00       	call   170f4 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x384>	170f0: R_X86_64_PLT32	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_-0x4
   170f4:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
   170f9:	48 8d 44 24 48       	lea    0x48(%rsp),%rax
   170fe:	48 39 c7             	cmp    %rax,%rdi
   17101:	74 0d                	je     17110 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3a0>
   17103:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
   17108:	48 ff c6             	inc    %rsi
   1710b:	e8 00 00 00 00       	call   17110 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3a0>	1710c: R_X86_64_PLT32	_ZdlPvm-0x4
   17110:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
   17115:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
   1711a:	48 39 c7             	cmp    %rax,%rdi
   1711d:	74 0d                	je     1712c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3bc>
   1711f:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
   17124:	48 ff c6             	inc    %rsi
   17127:	e8 00 00 00 00       	call   1712c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3bc>	17128: R_X86_64_PLT32	_ZdlPvm-0x4
   1712c:	e8 00 00 00 00       	call   17131 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3c1>	1712d: R_X86_64_PLT32	__cxa_end_catch-0x4
   17131:	31 c0                	xor    %eax,%eax
   17133:	e9 22 ff ff ff       	jmp    1705a <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x2ea>
   17138:	49 89 c7             	mov    %rax,%r15
   1713b:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
   17140:	48 8d 44 24 20       	lea    0x20(%rsp),%rax
   17145:	48 39 c7             	cmp    %rax,%rdi
   17148:	74 12                	je     1715c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3ec>
   1714a:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
   1714f:	48 ff c6             	inc    %rsi
   17152:	e8 00 00 00 00       	call   17157 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3e7>	17153: R_X86_64_PLT32	_ZdlPvm-0x4
   17157:	eb 03                	jmp    1715c <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3ec>
   17159:	49 89 c7             	mov    %rax,%r15
   1715c:	e8 00 00 00 00       	call   17161 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f1>	1715d: R_X86_64_PLT32	__cxa_end_catch-0x4
   17161:	4c 89 ff             	mov    %r15,%rdi
   17164:	e8 00 00 00 00       	call   17169 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x3f9>	17165: R_X86_64_PLT32	_Unwind_Resume-0x4
   17169:	48 89 c7             	mov    %rax,%rdi
   1716c:	e8 00 00 00 00       	call   17171 <_ZN6strata4core7MtpPair12capture_tailEiRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE+0x401>	1716d: R_X86_64_PLT32	__clang_call_terminate-0x4
