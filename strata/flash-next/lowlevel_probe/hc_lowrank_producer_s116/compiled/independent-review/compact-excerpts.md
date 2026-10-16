# Compact actual instruction appendix

Positions are instruction byte offsets; exact lines from pinned ASM. Full streams remain external.

## new_reduce

```text
15200         add (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r34.0<1;1,0>:f   {Compacted,$12.dst}
15208         add (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r38.0<1;1,0>:f   {Compacted,$13.dst}
15216         add (32|M0)              acc2.0<1>:f   acc2.0<1;1,0>:f   r36.0<1;1,0>:f   {Compacted,$14.dst}
15224         add (32|M0)              r30.0<1>:f    acc2.0<1;1,0>:f   r40.0<1;1,0>:f   {Compacted,$15.dst}
15232 (f3.0)  goto (32|M0)                         L16208                  L16208                
16240         add (32|M0)              acc0.0<1>:f   r30.0<1;1,0>:f    0.0:f               {Compacted}
16248         mov (16|M0)              r11.0<2>:ud   r14.0<1;1,0>:ud                  {Compacted,I@1}
16256         mul (32|M0)              r36.0<1>:f    acc0.0<1;1,0>:f   0.25:f               {Compacted}
16296         mul (32|M0)              r20.0<1>:f    r36.0<1;1,0>:f    -1.442695e+00:f               {A@1}
16312         add (16|M0)              r32.0<1>:q    r14.0<1;1,0>:q    r27.3<0;1,0>:q   {Compacted,$0.src}
16320         math.exp (32|M0)         r14.0<1>:f    r20.0<1;1,0>:f                   {A@1}
16360         add (32|M0)              r38.0<1>:f    r14.0<1;1,0>:f    1.0:f               {Compacted,M@1}
16368         add (16|M16)             r34.0<1>:q    r11.0<1;1,0>:q    r27.3<0;1,0>:q   {I@1}
16384         send.ugm (32|M0)         null     r32  r30:2  0x0            0x08000584           {I@1,$1} // wr:4+2, rd:0; store.ugm.d32.a64
16400 (W)     mov (1|M0)               f1.0<1>:ud    0xFFFFFFFF:ud                             
16416         sync.nop                             null                             {A@1}
16432         math.invm (16|M0) (eo)f1.0  r30.mme0:f  r36.nomme<1;1,0>:f  r38.nomme<1;1,0>:f {$1.src}
16592         join (32|M0)                         L17664                                
16608 (W)     mov (1|M0)               f1.0<1>:ud    0xFFFFFFFF:ud                             
16624         math.invm (16|M16) (eo)f1.0  r31.mme0:f  r37.nomme<1;1,0>:f  r39.nomme<1;1,0>:f {I@1}
16640 (f1.0)  goto (32|M0)                         L16768                  L16768                
16784         cmp (32|M0)   (lt)f1.0   null<1>:ud    r28.0<1;1,0>:ud   0x510:uw             
16800         mov (32|M0)              r20.0<1>:f    r30.0<1;1,0>:f                   {F@1}
16816 (f1.0)  goto (32|M0)                         L17232                  L17232                
17632         add (16|M0)              r7.0<1>:q     r11.0<1;1,0>:q    r118.0<1;1,0>:q  {Compacted,I@3}
17640         add (16|M16)             r9.0<1>:q     r5.0<1;1,0>:q     r116.0<1;1,0>:q  {Compacted,I@3}
17648         send.ugm (32|M0)         null     r7  r20:2  0x0            0x08000584           {A@1,$2} // wr:4+2, rd:0; store.ugm.d32.a64
```

## new_up

```text
1568         send.ugm (32|M0)         r46      r38  null:0  0x0            0x08200580           {I@1,$9} // wr:4+0, rd:2; load.ugm.d32.a64
1584         shl (32|M0)              r32.0<1>:d    r42.0<1;1,0>:d    2:w               {Compacted}
1592         mul (32|M0)              r38.0<1>:d    r44.0<1;1,0>:d    1280:w               {Compacted,$9.src}
1600         add3 (32|M0)             r30.0<1>:d    r38.0<1;0>:d      r32.0<1;0>:d      2048:w               {I@1}
1616         send.slm (32|M0)         null     r30  r46:2  0x0            0x04000504           {I@1,$9} // wr:2+2, rd:0; store.slm.d32.a32
1728 (W)     send.slm (1|M0)          r28      r27  null:0  0x0            0x0210001F           {F@1,$8} // wr:1+0, rd:1; fence.slm.none.group
1744 (W)     mov (8|M0)               null<1>:ud    r28.0<1;1,0>:ud                  {$8.dst}
1760 (W)     send.ugm (1|M0)          r28      r27  null:0  0x0            0x0210001F           {I@1,$10} // wr:1+0, rd:1; fence.ugm.none.group
1776 (W)     mov (1|M0)               r5.2<1>:f     3.58732e-43:f                              
1792 (W)     mov (2|M0)               r5.10<1>:ub   r27.11<0;1,0>:ub                 {F@1}
1808 (W)     mov (8|M0)               null<1>:ud    r28.0<1;1,0>:ud                  {$10.dst}
1824 (W)     send.gtwy (1|M0)         null     r5  null:0  0x0            0x02000004           {I@2,$11} // wr:1+0, rd:0; signal barrier
1840 (W)     sync.bar                             0x0                                
6048         mad (32|M0)              acc0.0<1>:f   r4.0<0;0>:f       r38.0<1;0>:f      r92.0<1>:f       {$3.dst}
6064         shl (32|M0)              r88.0<1>:d    r48.0<1;1,0>:d    16:w               {Compacted}
6072         mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r40.0<1;0>:f      r90.0<1>:f       {Compacted,I@2}
6288         mad (32|M0)              r17.0<1>:f    acc0.0<1;0>:f     r44.0<1;0>:f      r94.0<1>:f      
6568         mad (32|M0)              acc0.0<1>:f   acc0.0<1;0>:f     r44.0<1;0>:f      r38.0<1>:f       {Compacted,I@1}
6576         add (32|M0)              r17.0<1>:f    r17.0<1;1,0>:f    acc0.0<1;1,0>:f 
6752         add (32|M0)              r38.0<1>:f    r17.0<1;1,0>:f    r32.0<1;1,0>:f   {Compacted,I@1}
6808         add (32|M0)              r32.0<1>:f    r38.0<1;1,0>:f    r17.0<1;1,0>:f   {Compacted,I@1}
6912         add (32|M0)              r38.0<1>:f    r32.0<1;1,0>:f    r17.0<1;1,0>:f   {I@1}
6992         add (32|M0)              r32.0<1>:f    r38.0<1;1,0>:f    r17.0<1;1,0>:f   {Compacted,I@1}
7016         add (32|M0)              acc0.0<1>:f   r32.0<1;1,0>:f    r17.0<1;1,0>:f   {Compacted,I@1}
7024 (f3.0)  sel (32|M0)              r32.0<1>:f    acc0.0<1;1,0>:f   0.0:f              
13440         mul (32|M0)              acc0.0<1>:f   r30.0<1;1,0>:f    2.0:f               {Compacted,F@1}
13448         shl (16|M0)              r14.0<1>:q    r116.0<1;1,0>:q   2:w               {Compacted}
13456         shl (16|M16)             r17.0<1>:q    r114.0<1;1,0>:q   2:w               {Compacted}
13464         sync.nop                             null                             {Compacted,$2.dst}
13472         mad (32|M0)              r98.0<1>:f    r98.0<1;0>:f      acc0.0<1;0>:f     r112.0<1>:f      {Compacted,$14.dst}
13536         mul (32|M0)              r20.0<1>:f    r32.0<1;1,0>:f    -1.442695e+00:f              
13552         math.exp (32|M0)         r14.0<1>:f    r20.0<1;1,0>:f                   {F@1}
13568         sync.nop                             null                             {Compacted,$15.dst}
13576         mul (32|M0)              acc0.0<1>:f   r98.0<1;1,0>:f    r118.0<1;1,0>:f  {Compacted,$14.dst}
13584         sync.nop                             null                             {Compacted,M@1}
13592         add (32|M0)              r40.0<1>:f    r14.0<1;1,0>:f    1.0:f               {Compacted,$13.dst}
13600 (W)     mov (32|M0)              r17.0<1>:f    1.0:f                              
13616         mul (32|M0)              r42.0<1>:f    acc0.0<1;1,0>:f   r120.0<1;1,0>:f  {$0.dst}
14064         mul (32|M0)              r32.0<1>:f    r42.0<1;1,0>:f    r30.0<1;1,0>:f   {Compacted}
14072         or (32|M0)               r14.0<1>:d    r28.0<1;1,0>:d    r20.0<1;1,0>:d   {Compacted,I@1}
14080         send.slm (32|M0)         null     r14  r32:2  0x0            0x04000504           {A@1,$4} // wr:2+2, rd:0; store.slm.d32.a32
81256         add (32|M0)              acc0.0<1>:f   r18.0<1;1,0>:f    0.0:f               {Compacted,$8.dst}
81264         add (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r14.0<1;1,0>:f   {Compacted,$9.dst}
81272         add (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r10.0<1;1,0>:f   {Compacted,$10.dst}
81280         cmp (32|M0)   (lt)f3.0   null<1>:ud    r36.0<1;1,0>:ud   0x40:uw             
81296         add (32|M0)              acc0.0<1>:f   acc0.0<1;1,0>:f   r2.0<1;1,0>:f    {Compacted,$11.dst}
81304         mul (32|M0)              r12.0<1>:f    acc0.0<1;1,0>:f   0.25:f               {Compacted}
82080         send.ugm (32|M0)         null     r8  r12:2  0x0            0x08000584           {A@1,$7} // wr:4+2, rd:0; store.ugm.d32.a64
```

## old_up

```text
1056         send.ugm (32|M0)         r38      r30  null:0  0x0            0x08200580           {I@1,$10} // wr:4+0, rd:2; load.ugm.d32.a64
1072         add (32|M0)              r28.0<1>:f    r38.0<1;1,0>:f    0.0:f               {$10.dst}
1720         mul (32|M0)              r42.0<1>:f    r28.0<1;1,0>:f    0.25:f               {Compacted,F@1}
1728         mul (32|M0)              r28.0<1>:f    r42.0<1;1,0>:f    -1.442695e+00:f               {F@1}
1744         math.exp (32|M0)         r30.0<1>:f    r28.0<1;1,0>:f                   {F@1}
1760         add (32|M0)              r44.0<1>:f    r30.0<1;1,0>:f    1.0:f               {M@1}
1792         math.invm (16|M0) (eo)f2.0  r34.mme0:f  r42.nomme<1;1,0>:f  r44.nomme<1;1,0>:f {A@1}
1968         math.invm (16|M16) (eo)f2.0  r35.mme0:f  r43.nomme<1;1,0>:f  r45.nomme<1;1,0>:f {I@1}
```

