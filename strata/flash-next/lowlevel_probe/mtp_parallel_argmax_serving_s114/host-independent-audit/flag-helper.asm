00000000000098f0 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv>:
    98f0:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 98f7 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x7>	98f3: R_X86_64_PC32	.bss+0xa4
    98f7:	84 c0                	test   %al,%al
    98f9:	74 08                	je     9903 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x13>
    98fb:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 9902 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x12>	98fe: R_X86_64_PC32	.bss+0x9c
    9902:	c3                   	ret
    9903:	50                   	push   %rax
    9904:	bf 00 00 00 00       	mov    $0x0,%edi	9905: R_X86_64_32	.bss+0xa8
    9909:	e8 00 00 00 00       	call   990e <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x1e>	990a: R_X86_64_PLT32	__cxa_guard_acquire-0x4
    990e:	85 c0                	test   %eax,%eax
    9910:	74 2f                	je     9941 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x51>
    9912:	bf 00 00 00 00       	mov    $0x0,%edi	9913: R_X86_64_32	.rodata.str1.1+0x1dc3
    9917:	e8 00 00 00 00       	call   991c <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x2c>	9918: R_X86_64_PLT32	getenv-0x4
    991c:	48 85 c0             	test   %rax,%rax
    991f:	74 0e                	je     992f <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x3f>
    9921:	80 38 31             	cmpb   $0x31,(%rax)
    9924:	75 09                	jne    992f <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x3f>
    9926:	80 78 01 00          	cmpb   $0x0,0x1(%rax)
    992a:	0f 94 c0             	sete   %al
    992d:	eb 02                	jmp    9931 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x41>
    992f:	31 c0                	xor    %eax,%eax
    9931:	88 05 00 00 00 00    	mov    %al,0x0(%rip)        # 9937 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x47>	9933: R_X86_64_PC32	.bss+0x9c
    9937:	bf 00 00 00 00       	mov    $0x0,%edi	9938: R_X86_64_32	.bss+0xa8
    993c:	e8 00 00 00 00       	call   9941 <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x51>	993d: R_X86_64_PLT32	__cxa_guard_release-0x4
    9941:	48 83 c4 08          	add    $0x8,%rsp
    9945:	0f b6 05 00 00 00 00 	movzbl 0x0(%rip),%eax        # 994c <_ZN6strata4core12_GLOBAL__N_122mtp_parallel_argmax_onEv+0x5c>	9948: R_X86_64_PC32	.bss+0x9c
    994c:	c3                   	ret
