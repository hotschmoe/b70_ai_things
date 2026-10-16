L0:
(W)     mov (16|M0)              r127.0<1>:ud  0x0:ud                             
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud             
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw             
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x40:ud              {I@2}
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}
(W)     mad (1|M0)               r127.0<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0x40:uw              {I@1}
(W)     mov (8|M0)               r2.0<1>:ud    r1.0<1;1,0>:ud                  
(W)     send.ugm (1|M0)          r1       r127  null:0  0xFF000000            0x6219D500           {A@1,$0} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     and (1|M0)               r127.0<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud              {$0.src}
(W)     add (1|M0)               r127.0<1>:ud  r127.0<0;1,0>:ud  0x0:ud              {I@1}
(W)     send.ugm (1|M0)          r3       r127  null:0  0xFF000000            0x6219D500           {I@1,$1} // wr:1+0, rd:1; load.ugm.d32x16t.a32.ca.cc.bti[255]
(W)     mov (16|M0)              r38.0<1>:ud   r0.0<1;1,0>:ud                  
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x400004C0:ud              {A@1}
(W)     mul (1|M0)               acc0.0<1>:ud  r38.6<0;1,0>:ud   0x410:uw              {A@1}
(W)     asr (1|M0)               r3.3<1>:d     r2.6<0;1,0>:d     31:w               {$1.dst}
(W)     mach (1|M0)              r4.0<1>:d     r38.6<0;1,0>:ud   0x410:ud             
(W)     mul (1|M0)               acc0.0<1>:ud  r38.6<0;1,0>:ud   r2.12<0;1,0>:uw 
(W)     macl (1|M0)              r7.0<1>:ud    r38.6<0;1,0>:ud   r2.6<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:ud  r38.6<0;1,0>:ud   r2.12<0;1,0>:uw 
(W)     mach (1|M0)              r5.0<1>:d     r38.6<0;1,0>:ud   r2.6<0;1,0>:ud  
(W)     mul (1|M0)               acc0.0<1>:d   r38.6<0;1,0>:ud   r3.6<0;1,0>:uw   {I@6}
(W)     macl (1|M0)              r6.0<1>:d     r38.6<0;1,0>:ud   r3.3<0;1,0>:d   
(W)     mul (1|M0)               acc0.0<1>:d   r38.1<0;1,0>:d    r2.14<0;1,0>:uw 
(W)     mul (1|M0)               r2.8<1>:ud    r38.6<0;1,0>:ud   0x410:uw             
(W)     mov (1|M0)               r2.9<1>:d     r4.0<0;1,0>:d                    {I@7}
(W)     macl (1|M0)              r4.0<1>:d     r38.1<0;1,0>:d    r2.7<0;1,0>:d    {Compacted}
        mov (32|M0)              r22.0<1>:d    r1.0<1;1,0>:uw                   {$0.dst}
(W)     add (1|M0)               r39.0<1>:q    r2.4<0;1,0>:q     r2.1<0;1,0>:q    {Compacted,I@3}
(W)     add (1|M0)               r2.8<1>:d     r4.0<0;1,0>:d     r2.7<0;1,0>:d    {I@3}
        add (32|M0)              r50.0<1>:d    r4.0<0;1,0>:d     r22.0<1;1,0>:d   {Compacted,I@3}
(W)     sel (1|M0)    (lt)f0.0   r2.7<1>:d     r2.6<0;1,0>:d     r2.8<0;1,0>:d    {I@2}
(W)     add (1|M0)               r5.0<1>:d     r5.0<0;1,0>:d     r6.0<0;1,0>:d    {Compacted}
        cmp (32|M0)   (lt)f0.0   null<1>:d     r50.0<1;1,0>:d    r2.7<0;1,0>:d    {I@2}
(W)     add (1|M0)               r39.3<1>:q    r39.0<0;1,0>:q    16:w              
(W)     add (1|M0)               r39.2<1>:q    r39.0<0;1,0>:q    528:w               {Compacted}
(W)     mov (1|M0)               r3.3<1>:d     r5.0<0;1,0>:d                    {I@4}
(f0.0)  goto (32|M0)                         L592                  L592                
L552:
        mov (32|M0)              r26.0<1>:d    r2.6<0;1,0>:d                    {Compacted}
        mov (32|M0)              r24.0<1>:f    -inf:f                              
        goto (32|M0)                         L592                  L848                
L592:
        join (32|M0)                         L848                                
L608:
        mov (32|M0)              r24.0<1>:f    -inf:f                               {Compacted}
        mov (32|M0)              r26.0<1>:d    r2.6<0;1,0>:d                    {Compacted}
(W)     mov (1|M0)               r2.2<1>:d     r7.0<0;1,0>:d                   
(W)     mov (1|M0)               r2.3<1>:f     r3.3<0;1,0>:f                    {I@7}
L656:
        mov (16|M0)              r4.0<2>:ud    r50.0<1;1,0>:ud                  {Compacted}
        mov (16|M16)             r8.0<2>:ud    r51.0<1;1,0>:ud                  {Compacted}
        add (16|M0)              r6.0<1>:q     r2.1<0;1,0>:q     r4.0<2;1,0>:d    {A@1}
        add (16|M16)             r10.0<1>:q    r2.1<0;1,0>:q     r8.0<2;1,0>:d    {I@2}
        shl (16|M0)              r12.0<1>:q    r6.0<1;1,0>:q     2:w               {Compacted,I@2}
        shl (16|M16)             r14.0<1>:q    r10.0<1;1,0>:q    2:w               {Compacted,I@2}
        add (16|M0)              r16.0<1>:q    r12.0<1;1,0>:q    r2.0<0;1,0>:q    {Compacted,I@2}
        add (16|M16)             r18.0<1>:q    r14.0<1;1,0>:q    r2.0<0;1,0>:q    {Compacted,I@2}
        send.ugm (32|M0)         r20      r16  null:0  0x0            0x08200580           {I@1,$2} // wr:4+0, rd:2; load.ugm.d32.a64
        cmp (32|M0)   (gt)f1.0   null<1>:f     r20.0<1;1,0>:f    r24.0<1;1,0>:f   {$2.dst}
(f1.0)  mov (32|M0)              r26.0<1>:d    r50.0<1;1,0>:d                  
        add (32|M0)              r50.0<1>:d    r50.0<1;1,0>:d    r3.6<0;1,0>:d   
        cmp (32|M0)   (lt)f2.0   null<1>:d     r50.0<1;1,0>:d    r2.7<0;1,0>:d    {I@1}
(f1.0)  mov (32|M0)              r24.0<1>:ud   r20.0<1;1,0>:ud                  {F@1}
(f2.0)  goto.b (32|M0)                       L848                  L656                
L848:
        join (32|M0)                         L864                                
L864:
(W)     mov (8|M0)               r54.0<1>:w    0x76543210:v                              
(W)     mov (32|M0)              r4.0<1>:f     r24.0<1;1,0>:f                   {I@4}
(W)     add (8|M0)               r54.8<1>:w    r54.0<1;1,0>:w    8:w               {I@1}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     add (16|M0)              r54.16<1>:w   r54.0<1;1,0>:w    16:w               {I@2}
        cmp (32|M0)   (lt)f1.0   null<2>:uw    r54.0<1;1,0>:uw   0x10:uw              {I@1}
(W)     mov (1|M0)               r39.3<1>:ud   f1.0<0;1,0>:ud                  
(W)     mov (32|M0)              r10.0<1>:f    r26.0<1;1,0>:f                  
(W)     mov (1|M0)               f0.0<1>:ud    r39.3<0;1,0>:ud                  {Compacted,I@1}
(W)     mov (1|M0)               f3.0<1>:ud    r39.3<0;1,0>:ud                  {Compacted}
        mov (16|M0)              r8.0<1>:f     r5.0<1;1,0>:f                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:f     r6.0<1;1,0>:f                    {Compacted}
        mov (16|M0)              r13.0<1>:f    r6.0<1;1,0>:f                    {Compacted}
        mov (16|M0)              r12.0<1>:d    r11.0<1;1,0>:d                   {Compacted,F@4}
(f0.0)  sel (32|M0)              r96.0<1>:f    r8.0<1;1,0>:f     r24.0<1;1,0>:f   {F@2}
(f3.0)  sel (32|M0)              r10.0<1>:d    r12.0<1;1,0>:d    r26.0<1;1,0>:d   {A@1}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r96.0<1;1,0>:f    r24.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L1176                  L1176                
L1104:
        cmp (32|M0)   (lt)f1.0   null<1>:d     r10.0<1;1,0>:d    r26.0<1;1,0>:d  
(f1.0)  cmp (32|M0)   (eq)f1.0   null<1>:f     r96.0<1;1,0>:f    r24.0<1;1,0>:f   {I@1}
(f1.0)  goto (32|M0)                         L1176                  L1176                
L1152:
        mov (32|M0)              r68.0<1>:ud   r24.0<1;1,0>:ud                  {Compacted}
        goto (32|M0)                         L1176                  L1216                
L1176:
        join (32|M0)                         L1216                                
L1192:
        mov (32|M0)              r24.0<1>:f    r96.0<1;1,0>:f                   {Compacted,I@3}
        mov (32|M0)              r26.0<1>:d    r10.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r68.0<1>:f    r96.0<1;1,0>:f                   {Compacted}
L1216:
        join (32|M0)                         L1232                                
L1232:
        cmp (32|M0)   (lt)f2.0   null<2>:uw    r54.0<1;1,0>:uw   0x18:uw             
(W)     mov (1|M0)               r39.2<1>:ud   f2.0<0;1,0>:ud                  
(W)     mov (32|M0)              r4.0<1>:f     r24.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (1|M0)               f1.0<1>:ud    r39.2<0;1,0>:ud                  {Compacted,I@2}
        mov (16|M0)              r8.0<1>:d     r4.8<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.8<1;1,0>:d                    {I@3}
(f1.0)  sel (32|M0)              r102.0<1>:f   r8.0<1;1,0>:f     r68.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r26.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
(W)     mov (1|M0)               f0.0<1>:ud    r39.2<0;1,0>:ud                  {Compacted}
        cmp (32|M0)   (gt)f2.0   null<1>:f     r102.0<1;1,0>:f   r68.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.8<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.8<1;1,0>:d                   {I@3}
(f0.0)  sel (32|M0)              r18.0<1>:d    r14.0<1;1,0>:d    r26.0<1;1,0>:d   {I@1}
(f2.0)  goto (32|M0)                         L1536                  L1536                
L1456:
        cmp (32|M0)   (lt)f0.0   null<1>:d     r18.0<1;1,0>:d    r26.0<1;1,0>:d   {I@2}
(f0.0)  cmp (32|M0)   (eq)f0.0   null<1>:f     r102.0<1;1,0>:f   r68.0<1;1,0>:f   {I@1}
(f0.0)  goto (32|M0)                         L1536                  L1536                
L1504:
        mov (32|M0)              r66.0<1>:ud   r68.0<1;1,0>:ud                 
        goto (32|M0)                         L1536                  L1584                
L1536:
        join (32|M0)                         L1584                                
L1552:
        mov (32|M0)              r24.0<1>:f    r102.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r26.0<1>:d    r18.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r66.0<1>:f    r102.0<1;1,0>:f                  {I@4}
L1584:
        join (32|M0)                         L1600                                
L1600:
(W)     mov (32|M0)              r4.0<1>:f     r24.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        cmp (32|M0)   (lt)f2.0   null<2>:uw    r54.0<1;1,0>:uw   0x1C:uw             
        mov (16|M0)              r8.0<1>:d     r4.4<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.4<1;1,0>:d                    {Compacted,I@3}
(f2.0)  sel (32|M0)              r100.0<1>:f   r8.0<1;1,0>:f     r66.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r26.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f1.0   null<1>:f     r100.0<1;1,0>:f   r66.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.4<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.4<1;1,0>:d                   {Compacted,I@2}
(f2.0)  sel (32|M0)              r16.0<1>:d    r14.0<1;1,0>:d    r26.0<1;1,0>:d   {I@1}
(f1.0)  goto (32|M0)                         L1856                  L1856                
L1776:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r16.0<1;1,0>:d    r26.0<1;1,0>:d   {I@2}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r100.0<1;1,0>:f   r66.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L1856                  L1856                
L1824:
        mov (32|M0)              r64.0<1>:ud   r66.0<1;1,0>:ud                 
        goto (32|M0)                         L1856                  L1904                
L1856:
        join (32|M0)                         L1904                                
L1872:
        mov (32|M0)              r24.0<1>:f    r100.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r26.0<1>:d    r16.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r64.0<1>:f    r100.0<1;1,0>:f                  {I@4}
L1904:
        join (32|M0)                         L1920                                
L1920:
(W)     mov (32|M0)              r4.0<1>:f     r24.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        cmp (32|M0)   (lt)f1.0   null<2>:uw    r54.0<1;1,0>:uw   0x1E:uw             
        mov (16|M0)              r8.0<1>:d     r4.2<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.2<1;1,0>:d                    {Compacted,I@3}
(f1.0)  sel (32|M0)              r98.0<1>:f    r8.0<1;1,0>:f     r64.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r26.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f0.0   null<1>:f     r98.0<1;1,0>:f    r64.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.2<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.2<1;1,0>:d                   {Compacted,I@2}
(f1.0)  sel (32|M0)              r12.0<1>:d    r14.0<1;1,0>:d    r26.0<1;1,0>:d   {I@1}
(f0.0)  goto (32|M0)                         L2176                  L2176                
L2096:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r12.0<1;1,0>:d    r26.0<1;1,0>:d   {I@2}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r98.0<1;1,0>:f    r64.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L2176                  L2176                
L2144:
        mov (32|M0)              r62.0<1>:ud   r64.0<1;1,0>:ud                 
        goto (32|M0)                         L2176                  L2224                
L2176:
        join (32|M0)                         L2224                                
L2192:
        mov (32|M0)              r24.0<1>:f    r98.0<1;1,0>:f                   {Compacted}
        mov (32|M0)              r26.0<1>:d    r12.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r62.0<1>:f    r98.0<1;1,0>:f                   {I@4}
L2224:
        join (32|M0)                         L2240                                
L2240:
(W)     mov (32|M0)              r4.0<1>:f     r24.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        cmp (32|M0)   (eq)f0.0   null<2>:w     r54.0<1;1,0>:w    31:w              
        mov (16|M0)              r8.0<1>:d     r4.1<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.1<1;1,0>:d                    {Compacted,I@3}
(f0.0)  sel (32|M0)              r24.0<1>:f    r62.0<1;1,0>:f    r8.0<1;1,0>:f    {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r26.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r24.0<1;1,0>:f    r62.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.1<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.1<1;1,0>:d                   {Compacted,I@2}
(f0.0)  sel (32|M0)              r8.0<1>:d     r26.0<1;1,0>:d    r14.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L2496                  L2496                
L2416:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r8.0<1;1,0>:d     r26.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r24.0<1;1,0>:f    r62.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L2496                  L2496                
L2464:
        mov (32|M0)              r24.0<1>:ud   r62.0<1;1,0>:ud                  {F@1}
        goto (32|M0)                         L2496                  L2528                
L2496:
        join (32|M0)                         L2528                                
L2512:
        mov (32|M0)              r26.0<1>:f    r8.0<1;1,0>:f                   
L2528:
        join (32|M0)                         L2544                                
L2544:
        mov (16|M0)              r4.0<4>:uw    r1.0<1;1,0>:uw                  
        mov (16|M16)             r6.0<4>:uw    r1.16<1;1,0>:uw                 
        and (32|M0)   (eq)f3.0   r60.0<1>:w    r1.0<1;1,0>:w     31:w              
        mov (16|M0)              r48.0<1>:q    r4.0<4;1,0>:uw                   {I@3}
        mov (16|M16)             r46.0<1>:q    r6.0<4;1,0>:uw                   {I@3}
(W)     mov (1|M0)               r39.9<1>:ud   f3.0<0;1,0>:ud                  
        shr (16|M0)              r36.0<1>:uq   r48.0<1;1,0>:uq   5:w               {I@3}
        shr (16|M16)             r34.0<1>:uq   r46.0<1;1,0>:uq   5:w               {I@3}
(~f3.0) goto (32|M0)                         L2752                  L2752                
L2688:
        mov (16|M0)              r4.0<1>:d     r36.0<2;1,0>:d                   {Compacted,I@3}
        mov (16|M16)             r5.0<1>:d     r34.0<2;1,0>:d                   {Compacted,I@3}
        shl (32|M0)              r6.0<1>:d     r4.0<1;1,0>:d     2:w               {I@1}
        send.slm (32|M0)         null     r6  r24:2  0x100000            0x04000504           {I@1,$3} // wr:2+2, rd:0; store.slm.d32.a32.flat[A+0x100]
        send.slm (32|M0)         null     r6  r26:2  0x0            0x04000504           {F@1,$4} // wr:2+2, rd:0; store.slm.d32.a32
L2752:
        join (32|M0)                         L2768                                
L2768:
(W)     send.slm (1|M0)          r4       r38  null:0  0x0            0x0210001F           {$5} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r4.0<1;1,0>:ud                   {$5.dst}
(W)     send.ugm (1|M0)          r5       r38  null:0  0x0            0x0210001F           {$6} // wr:1+0, rd:1; fence.ugm.none.group
(W)     mov (1|M0)               r6.2<1>:f     3.58732e-43:f                               {$4.src}
(W)     mov (2|M0)               r6.10<1>:ub   r38.11<0;1,0>:ub                 {F@1}
(W)     mov (8|M0)               null<1>:ud    r5.0<1;1,0>:ud                   {$6.dst}
(W)     send.gtwy (1|M0)         null     r6  null:0  0x0            0x02000004           {I@2,$7} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             0x0                                
        cmp (32|M0)   (lt)f3.0   null<2>:uw    r1.0<1;1,0>:uw    0x20:uw             
        mov (32|M0)              r30.0<1>:d    r60.0<1;1,0>:uw                 
(W)     mov (1|M0)               r39.8<1>:ud   f3.0<0;1,0>:ud                  
(f3.0)  goto (32|M0)                         L2992                  L2992                
L2960:
        mov (32|M0)              r28.0<1>:d    0:w                              
        goto (32|M0)                         L2992                  L5616                
L2992:
        join (32|M0)                         L5616                                
L3008:
(W)     shr (1|M0)               r2.8<1>:ud    r3.6<0;1,0>:ud    5:w              
        cmp (32|M0)   (gt)f3.0   null<1>:ud    r2.8<0;1,0>:ud    r30.0<1;1,0>:ud  {I@1}
(f3.0)  goto (32|M0)                         L3088                  L3088                
L3056:
        mov (32|M0)              r44.0<1>:f    -inf:f                               {Compacted}
        mov (32|M0)              r42.0<1>:d    r2.6<0;1,0>:d                    {Compacted}
        goto (32|M0)                         L3088                  L3152                
L3088:
        join (32|M0)                         L3152                                
L3104:
        shl (32|M0)              r4.0<1>:d     r22.0<1;1,0>:d    2:w              
        send.slm (32|M0)         r44      r4  null:0  0x100000            0x04200500           {A@1,$8} // wr:2+0, rd:2; load.slm.d32.a32.flat[A+0x100]
        send.slm (32|M0)         r42      r4  null:0  0x0            0x04200500           {$9} // wr:2+0, rd:2; load.slm.d32.a32
L3152:
        join (32|M0)                         L5616                                
L3168:
        sync.nop                             null                             {$9.src}
(W)     mov (32|M0)              r4.0<1>:f     r44.0<1;1,0>:f                   {$8.dst}
(W)     mov (32|M0)              r6.0<1>:d     0:w                               {$7.src}
(W)     mov (1|M0)               f3.0<1>:ud    r39.3<0;1,0>:ud                  {Compacted}
(W)     mov (32|M0)              r10.0<1>:f    r42.0<1;1,0>:f                   {$9.dst}
        mov (16|M0)              r8.0<1>:d     r5.0<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r6.0<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r13.0<1>:d    r6.0<1;1,0>:d                    {Compacted}
        mov (16|M0)              r12.0<1>:d    r11.0<1;1,0>:d                   {F@1}
(f3.0)  sel (32|M0)              r112.0<1>:f   r8.0<1;1,0>:f     r44.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r52.0<1>:d    r12.0<1;1,0>:d    r42.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r112.0<1;1,0>:f   r44.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L3416                  L3416                
L3344:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r52.0<1;1,0>:d    r42.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r112.0<1;1,0>:f   r44.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L3416                  L3416                
L3392:
        mov (32|M0)              r92.0<1>:ud   r44.0<1;1,0>:ud                  {Compacted}
        goto (32|M0)                         L3416                  L3456                
L3416:
        join (32|M0)                         L3456                                
L3432:
        mov (32|M0)              r44.0<1>:f    r112.0<1;1,0>:f                  {Compacted,I@3}
        mov (32|M0)              r42.0<1>:d    r52.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r92.0<1>:f    r112.0<1;1,0>:f                  {Compacted}
L3456:
        join (32|M0)                         L5616                                
L3472:
(W)     mov (32|M0)              r4.0<1>:f     r44.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (1|M0)               f3.0<1>:ud    r39.2<0;1,0>:ud                 
(W)     mov (32|M0)              r10.0<1>:f    r42.0<1;1,0>:f                   {I@4}
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.8<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.8<1;1,0>:d                    {Compacted,I@4}
        mov (16|M0)              r14.0<1>:d    r10.8<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.8<1;1,0>:d                   {Compacted,I@4}
(f3.0)  sel (32|M0)              r118.0<1>:f   r8.0<1;1,0>:f     r92.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r58.0<1>:d    r14.0<1;1,0>:d    r42.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r118.0<1;1,0>:f   r92.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L3728                  L3728                
L3648:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r58.0<1;1,0>:d    r42.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r118.0<1;1,0>:f   r92.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L3728                  L3728                
L3696:
        mov (32|M0)              r90.0<1>:ud   r92.0<1;1,0>:ud                 
        goto (32|M0)                         L3728                  L3776                
L3728:
        join (32|M0)                         L3776                                
L3744:
        mov (32|M0)              r44.0<1>:f    r118.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r42.0<1>:d    r58.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r90.0<1>:f    r118.0<1;1,0>:f                  {I@4}
L3776:
        join (32|M0)                         L5616                                
L3792:
(W)     mov (32|M0)              r4.0<1>:f     r44.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.4<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.4<1;1,0>:d                    {Compacted,I@2}
(f2.0)  sel (32|M0)              r116.0<1>:f   r8.0<1;1,0>:f     r90.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r42.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r116.0<1;1,0>:f   r90.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.4<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.4<1;1,0>:d                   {Compacted,I@2}
(f2.0)  sel (32|M0)              r56.0<1>:d    r14.0<1;1,0>:d    r42.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L4032                  L4032                
L3952:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r56.0<1;1,0>:d    r42.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r116.0<1;1,0>:f   r90.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L4032                  L4032                
L4000:
        mov (32|M0)              r88.0<1>:ud   r90.0<1;1,0>:ud                 
        goto (32|M0)                         L4032                  L4080                
L4032:
        join (32|M0)                         L4080                                
L4048:
        mov (32|M0)              r44.0<1>:f    r116.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r42.0<1>:d    r56.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r88.0<1>:f    r116.0<1;1,0>:f                  {I@4}
L4080:
        join (32|M0)                         L5616                                
L4096:
(W)     mov (32|M0)              r4.0<1>:f     r44.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.2<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.2<1;1,0>:d                    {Compacted,I@2}
(f1.0)  sel (32|M0)              r114.0<1>:f   r8.0<1;1,0>:f     r88.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r42.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r114.0<1;1,0>:f   r88.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.2<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.2<1;1,0>:d                   {Compacted,I@2}
(f1.0)  sel (32|M0)              r54.0<1>:d    r14.0<1;1,0>:d    r42.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L4328                  L4328                
L4256:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r54.0<1;1,0>:d    r42.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r114.0<1;1,0>:f   r88.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L4328                  L4328                
L4304:
        mov (32|M0)              r86.0<1>:ud   r88.0<1;1,0>:ud                  {Compacted}
        goto (32|M0)                         L4328                  L4368                
L4328:
        join (32|M0)                         L4368                                
L4344:
        mov (32|M0)              r44.0<1>:f    r114.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r42.0<1>:d    r54.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r86.0<1>:f    r114.0<1;1,0>:f                  {Compacted,I@4}
L4368:
        join (32|M0)                         L5616                                
L4384:
(W)     mov (32|M0)              r4.0<1>:f     r44.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.1<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.1<1;1,0>:d                    {Compacted,I@2}
(f0.0)  sel (32|M0)              r44.0<1>:f    r86.0<1;1,0>:f    r8.0<1;1,0>:f    {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r42.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r44.0<1;1,0>:f    r86.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.1<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.1<1;1,0>:d                   {Compacted,I@2}
(f0.0)  sel (32|M0)              r50.0<1>:d    r42.0<1;1,0>:d    r14.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L4624                  L4624                
L4544:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r50.0<1;1,0>:d    r42.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r44.0<1;1,0>:f    r86.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L4624                  L4624                
L4592:
        mov (32|M0)              r44.0<1>:ud   r86.0<1;1,0>:ud                  {F@1}
        goto (32|M0)                         L4624                  L4656                
L4624:
        join (32|M0)                         L4656                                
L4640:
        mov (32|M0)              r42.0<1>:f    r50.0<1;1,0>:f                  
L4656:
        join (32|M0)                         L5616                                
L4672:
        cmp (32|M0)   (eq)f3.0   null<2>:w     r1.0<1;1,0>:w     0:w              
(f3.0)  goto (32|M0)                         L4752                  L4752                
L4704:
(W)     mov (1|M0)               r2.16<1>:hf   -qnan(0x1FF):hf                             
(f3.0)  sel (32|M0)              r28.0<1>:d    r2.16<0;1,0>:w    0:w               {F@1}
        goto (32|M0)                         L4752                  L5616                
L4752:
        join (32|M0)                         L5616                                
L4768:
(W)     shl (1|M0)               r2.4<1>:q     r38.1<0;1,0>:ud   2:w              
(W)     add (1|M0)               r4.0<1>:q     r39.3<0;1,0>:q    r2.4<0;1,0>:q    {Compacted,I@1}
(W)     add (1|M0)               r6.0<1>:q     r39.2<0;1,0>:q    r2.4<0;1,0>:q    {Compacted}
        mov (16|M0)              r7.0<2>:f     r4.0<0;1,0>:f                    {I@2}
        mov (16|M16)             r9.0<2>:f     r4.0<0;1,0>:f                   
        mov (16|M0)              r7.1<2>:f     r4.1<0;1,0>:f                   
        mov (16|M16)             r9.1<2>:f     r4.1<0;1,0>:f                   
        mov (16|M0)              r11.0<2>:f    r6.0<0;1,0>:f                    {I@1}
        mov (16|M16)             r13.0<2>:f    r6.0<0;1,0>:f                   
        mov (16|M0)              r11.1<2>:f    r6.1<0;1,0>:f                   
        mov (16|M16)             r13.1<2>:f    r6.1<0;1,0>:f                   
        send.ugm (32|M0)         null     r7  r44:2  0x0            0x08000584           {F@5,$10} // wr:4+2, rd:0; store.ugm.d32.a64
        send.ugm (32|M0)         null     r11  r42:2  0x0            0x08000584           {F@1,$11} // wr:4+2, rd:0; store.ugm.d32.a64
(W)     send.slm (1|M0)          r5       r38  null:0  0x0            0x0210001F           {$12} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r5.0<1;1,0>:ud                   {$12.dst}
(W)     send.ugm (1|M0)          r15      r38  null:0  0x0            0x0210261F           {$13} // wr:1+0, rd:1; fence.ugm.invalidate.gpu
(W)     mov (32|M0)              r16.0<1>:d    0:w                              
        mov (32|M0)              r16.0<1>:d    1:w                              
(W)     add (16|M0)              r16.1<2>:d    r16.0<2;1,0>:d    r16.1<2;1,0>:d   {I@1}
(W)     add (8|M0)               r16.2<4>:d    r16.2<4;1,0>:d    r16.1<4;1,0>:d   {I@1}
(W)     mov (1|M0)               r18.0<1>:uq   r39.0<0;1,0>:uq                 
(W)     add (8|M0)               r16.3<4>:d    r16.3<4;1,0>:d    r16.1<4;1,0>:d   {I@2}
(W)     add (4|M0)               r16.4<1>:d    r16.3<0;1,0>:d    r16.4<1;1,0>:d   {I@1}
(W)     add (4|M0)               r16.12<1>:d   r16.11<0;1,0>:d   r16.12<1;1,0>:d 
(W)     add (4|M0)               r17.4<1>:d    r17.3<0;1,0>:d    r17.4<1;1,0>:d  
(W)     add (4|M0)               r17.12<1>:d   r17.11<0;1,0>:d   r17.12<1;1,0>:d 
(W)     add (8|M0)               r16.8<1>:d    r16.7<0;1,0>:d    r16.8<1;1,0>:d   {I@3}
(W)     add (8|M0)               r17.8<1>:d    r17.7<0;1,0>:d    r17.8<1;1,0>:d   {I@2}
(W)     add (16|M0)              r17.0<1>:d    r16.15<0;1,0>:d   r17.0<1;1,0>:d   {I@1}
(W)     mov (1|M0)               r4.0<1>:d     r17.15<0;1,0>:d                  {I@1}
(W)     mov (8|M0)               null<1>:ud    r15.0<1;1,0>:ud                  {$13.dst}
(W)     send.ugm (1|M0)          r5       r18  r4:1  0x0            0x0410058C           {I@2,$14} // wr:2+1, rd:1; atomic_iadd.ugm.d32.a64
(W)     add (32|M0)              r16.0<1>:d    r16.0<1;1,0>:d    r5.0<0;1,0>:ud   {$14.dst}
        add (32|M0)              r6.0<1>:d     r16.0<1;1,0>:d    -1:w               {Compacted,@1,$10.src}
(W)     send.slm (1|M0)          r8       r38  null:0  0x0            0x0210001F           {$15} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r8.0<1;1,0>:ud                   {Compacted,$15.dst}
(W)     send.ugm (1|M0)          r9       r38  null:0  0x0            0x0210261F           {$0} // wr:1+0, rd:1; fence.ugm.invalidate.gpu
(W)     add (1|M0)               r2.9<1>:d     r3.0<0;1,0>:d     -1:w              
(W)     mov (1|M0)               r2.8<1>:ud    0x0:uw                             
        cmp (32|M0)   (eq)f3.0   null<1>:d     r6.0<1;1,0>:d     r2.9<0;1,0>:d    {I@2}
(W)     mov (1|M0)               r3.3<1>:ud    f3.0<0;1,0>:ud                  
(W)     mov (1|M0)               f3.0<1>:ud    r2.8<0;1,0>:ud                   {Compacted,I@3}
        cmp (32|M0)   (eq)f3.0   null<2>:uw    r1.0<0;1,0>:uw    r1.0<0;1,0>:uw  
(W)     mov (1|M0)               r2.8<1>:ud    f3.0<0;1,0>:ud                  
(W)     mov (1|M0)               f3.0<1>:ud    r3.3<0;1,0>:ud                   {Compacted,I@4}
(W)     mov (1|M0)               r2.20<1>:hf   -qnan(0x1FF):hf                             
(W)     fbl (1|M0)               r5.0<1>:ud    r2.8<0;1,0>:ud                   {Compacted,I@2}
        mov (32|M0)              r28.0<1>:d    -1:w                               {Compacted}
(f3.0)  sel (32|M0)              r4.0<1>:w     r2.20<0;1,0>:w    0:w               {F@1}
(W)     add (1|M0)               a0.0<1>:uw    r5.0<0;1,0>:uw    0x40:uw              {A@1}
        mov (32|M0)              r8.0<2>:b     -r4.0<1;1,0>:w                   {I@2}
        mov (32|M0)              r1.0<1>:b     r8.0<2;1,0>:b                    {I@1}
(W)     mov (8|M0)               null<1>:ud    r9.0<1;1,0>:ud                   {$0.dst}
(W)     mov (1|M0)               r9.0<2>:b     r[a0.0]<0;1,0>:b                 {I@2}
(W)     mov (1|M0)               r10.0<1>:ud   0x200:uw                             
(W)     mov (1|M0)               r6.0<1>:ud    r9.0<0;1,0>:ub                   {I@2}
(W)     send.slm (1|M0)          null     r10  r6:1  0x0            0x02000904           {I@1,$1} // wr:1+1, rd:0; store.slm.d8u32.a32
L5616:
        join (32|M0)                         L5632                                
L5632:
        sync.nop                             null                             {Compacted,$9.src}
(W)     send.slm (1|M0)          r4       r38  null:0  0x0            0x0210001F           {$2} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r4.0<1;1,0>:ud                   {Compacted,$2.dst}
(W)     send.ugm (1|M0)          r5       r38  null:0  0x0            0x0210001F           {$4} // wr:1+0, rd:1; fence.ugm.none.group
        sync.nop                             null                             {$1.src}
(W)     mov (1|M0)               r6.2<1>:f     3.58732e-43:f                               {$7.src}
(W)     mov (2|M0)               r6.10<1>:ub   r38.11<0;1,0>:ub                 {F@1}
(W)     mov (8|M0)               null<1>:ud    r5.0<1;1,0>:ud                   {$4.dst}
(W)     send.gtwy (1|M0)         null     r6  null:0  0x0            0x02000004           {I@2,$5} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             0x0                                
(W)     mov (1|M0)               r7.0<1>:ud    0x200:uw                             
(W)     send.slm (1|M0)          r8       r7  null:0  0x0            0x02100900           {I@1,$6} // wr:1+0, rd:1; load.slm.d8u32.a32
(W)     mov (1|M0)               r9.0<2>:b     r8.0<0;1,0>:b                    {$6.dst}
(W)     mov (1|M0)               r2.16<1>:w    r9.0<0;1,0>:ub                   {I@1}
(W)     cmp (32|M0)   (eq)f3.0   null<2>:w     r2.16<0;1,0>:w    0:w               {I@1}
(W&f3.0) jmpi                                L9688                                
L5872:
(W)     send.slm (1|M0)          r4       r38  null:0  0x0            0x0210001F           {$7} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r4.0<1;1,0>:ud                   {Compacted,$7.dst}
(W)     send.ugm (1|M0)          r5       r38  null:0  0x0            0x0210261F           {$10} // wr:1+0, rd:1; fence.ugm.invalidate.gpu
        cmp (32|M0)   (gt)f3.0   null<1>:ud    r3.0<0;1,0>:ud    r22.0<1;1,0>:ud 
(W)     mov (8|M0)               null<1>:ud    r5.0<1;1,0>:ud                   {Compacted,$10.dst}
(f3.0)  goto (32|M0)                         L5984                  L5984                
L5952:
        mov (32|M0)              r32.0<1>:d    r2.6<0;1,0>:d                    {Compacted}
        mov (32|M0)              r40.0<1>:f    -inf:f                               {Compacted}
        goto (32|M0)                         L5984                  L6080                
L5984:
        join (32|M0)                         L6080                                
L6000:
        shl (16|M0)              r4.0<1>:q     r48.0<1;1,0>:q    2:w               {Compacted}
        shl (16|M16)             r6.0<1>:q     r46.0<1;1,0>:q    2:w               {Compacted,$5.src}
        add (16|M0)              r8.0<1>:q     r39.3<0;1,0>:q    r4.0<1;1,0>:q    {Compacted,I@2}
        add (16|M0)              r12.0<1>:q    r39.2<0;1,0>:q    r4.0<1;1,0>:q    {Compacted,$11.src}
        add (16|M16)             r10.0<1>:q    r39.3<0;1,0>:q    r6.0<1;1,0>:q    {Compacted,I@3}
        add (16|M16)             r14.0<1>:q    r39.2<0;1,0>:q    r6.0<1;1,0>:q    {Compacted}
        send.ugm (32|M0)         r40      r8  null:0  0x0            0x08200580           {A@1,$12} // wr:4+0, rd:2; load.ugm.d32.a64
        send.ugm (32|M0)         r32      r12  null:0  0x0            0x08200580           {I@1,$13} // wr:4+0, rd:2; load.ugm.d32.a64
L6080:
        join (32|M0)                         L6096                                
L6096:
(W)     mov (32|M0)              r4.0<1>:f     r40.0<1;1,0>:f                   {$12.dst}
(W)     mov (32|M0)              r6.0<1>:d     0:w                               {$5.src}
(W)     mov (1|M0)               f3.0<1>:ud    r39.3<0;1,0>:ud                  {Compacted}
        sync.nop                             null                             {Compacted,$11.src}
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {$13.dst}
        mov (16|M0)              r8.0<1>:d     r5.0<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r6.0<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r13.0<1>:d    r6.0<1;1,0>:d                    {Compacted}
        mov (16|M0)              r12.0<1>:d    r11.0<1;1,0>:d                   {Compacted,F@1}
(f3.0)  sel (32|M0)              r104.0<1>:f   r8.0<1;1,0>:f     r40.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r24.0<1>:d    r12.0<1;1,0>:d    r32.0<1;1,0>:d   {@1,$3.src}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r104.0<1;1,0>:f   r40.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L6336                  L6336                
L6256:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r24.0<1;1,0>:d    r32.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r104.0<1;1,0>:f   r40.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L6336                  L6336                
L6304:
        mov (32|M0)              r76.0<1>:ud   r40.0<1;1,0>:ud                 
        goto (32|M0)                         L6336                  L6384                
L6336:
        join (32|M0)                         L6384                                
L6352:
        mov (32|M0)              r40.0<1>:f    r104.0<1;1,0>:f                  {Compacted,I@3}
        mov (32|M0)              r32.0<1>:d    r24.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r76.0<1>:f    r104.0<1;1,0>:f                 
L6384:
        join (32|M0)                         L6400                                
L6400:
(W)     mov (32|M0)              r4.0<1>:f     r40.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (1|M0)               f3.0<1>:ud    r39.2<0;1,0>:ud                 
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {I@4}
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.8<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.8<1;1,0>:d                    {Compacted,I@4}
        mov (16|M0)              r14.0<1>:d    r10.8<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.8<1;1,0>:d                   {Compacted,I@4}
(f3.0)  sel (32|M0)              r110.0<1>:f   r8.0<1;1,0>:f     r76.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r44.0<1>:d    r14.0<1;1,0>:d    r32.0<1;1,0>:d   {@1,$8.dst}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r110.0<1;1,0>:f   r76.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L6656                  L6656                
L6576:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r44.0<1;1,0>:d    r32.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r110.0<1;1,0>:f   r76.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L6656                  L6656                
L6624:
        mov (32|M0)              r74.0<1>:ud   r76.0<1;1,0>:ud                 
        goto (32|M0)                         L6656                  L6704                
L6656:
        join (32|M0)                         L6704                                
L6672:
        mov (32|M0)              r40.0<1>:f    r110.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r32.0<1>:d    r44.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r74.0<1>:f    r110.0<1;1,0>:f                  {I@4}
L6704:
        join (32|M0)                         L6720                                
L6720:
(W)     mov (32|M0)              r4.0<1>:f     r40.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.4<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.4<1;1,0>:d                    {Compacted,I@2}
(f2.0)  sel (32|M0)              r108.0<1>:f   r8.0<1;1,0>:f     r74.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r108.0<1;1,0>:f   r74.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.4<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.4<1;1,0>:d                   {Compacted,I@2}
(f2.0)  sel (32|M0)              r42.0<1>:d    r14.0<1;1,0>:d    r32.0<1;1,0>:d   {@1,$9.dst}
(f3.0)  goto (32|M0)                         L6952                  L6952                
L6880:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r42.0<1;1,0>:d    r32.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r108.0<1;1,0>:f   r74.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L6952                  L6952                
L6928:
        mov (32|M0)              r72.0<1>:ud   r74.0<1;1,0>:ud                  {Compacted}
        goto (32|M0)                         L6952                  L6992                
L6952:
        join (32|M0)                         L6992                                
L6968:
        mov (32|M0)              r40.0<1>:f    r108.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r32.0<1>:d    r42.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r72.0<1>:f    r108.0<1;1,0>:f                  {Compacted,I@4}
L6992:
        join (32|M0)                         L7008                                
L7008:
(W)     mov (32|M0)              r4.0<1>:f     r40.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.2<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.2<1;1,0>:d                    {Compacted,I@2}
(f1.0)  sel (32|M0)              r106.0<1>:f   r8.0<1;1,0>:f     r72.0<1;1,0>:f   {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r106.0<1;1,0>:f   r72.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.2<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.2<1;1,0>:d                   {Compacted,I@2}
(f1.0)  sel (32|M0)              r26.0<1>:d    r14.0<1;1,0>:d    r32.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L7248                  L7248                
L7168:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r26.0<1;1,0>:d    r32.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r106.0<1;1,0>:f   r72.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L7248                  L7248                
L7216:
        mov (32|M0)              r70.0<1>:ud   r72.0<1;1,0>:ud                 
        goto (32|M0)                         L7248                  L7296                
L7248:
        join (32|M0)                         L7296                                
L7264:
        mov (32|M0)              r40.0<1>:f    r106.0<1;1,0>:f                  {Compacted}
        mov (32|M0)              r32.0<1>:d    r26.0<1;1,0>:d                   {Compacted}
        mov (32|M0)              r70.0<1>:f    r106.0<1;1,0>:f                  {I@4}
L7296:
        join (32|M0)                         L7312                                
L7312:
(W)     mov (32|M0)              r4.0<1>:f     r40.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
        mov (16|M0)              r8.0<1>:d     r4.1<1;1,0>:d                    {Compacted,F@1}
        mov (16|M0)              r9.0<1>:d     r5.1<1;1,0>:d                    {Compacted,I@2}
(f0.0)  sel (32|M0)              r40.0<1>:f    r70.0<1;1,0>:f    r8.0<1;1,0>:f    {I@1}
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        cmp (32|M0)   (gt)f3.0   null<1>:f     r40.0<1;1,0>:f    r70.0<1;1,0>:f   {F@2}
        mov (16|M0)              r14.0<1>:d    r10.1<1;1,0>:d                   {Compacted,F@2}
        mov (16|M0)              r15.0<1>:d    r11.1<1;1,0>:d                   {Compacted,I@2}
(f0.0)  sel (32|M0)              r20.0<1>:d    r32.0<1;1,0>:d    r14.0<1;1,0>:d   {I@1}
(f3.0)  goto (32|M0)                         L7552                  L7552                
L7472:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r20.0<1;1,0>:d    r32.0<1;1,0>:d   {A@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r40.0<1;1,0>:f    r70.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L7552                  L7552                
L7520:
        mov (32|M0)              r40.0<1>:ud   r70.0<1;1,0>:ud                  {F@1}
        goto (32|M0)                         L7552                  L7576                
L7552:
        join (32|M0)                         L7576                                
L7568:
        mov (32|M0)              r32.0<1>:f    r20.0<1;1,0>:f                   {Compacted}
L7576:
        join (32|M0)                         L7592                                
L7592:
(W)     mov (1|M0)               f3.0<1>:ud    r39.9<0;1,0>:ud                  {Compacted}
(~f3.0) goto (32|M0)                         L7680                  L7680                
L7616:
        mov (16|M0)              r4.0<1>:d     r36.0<2;1,0>:d                   {Compacted}
        mov (16|M16)             r5.0<1>:d     r34.0<2;1,0>:d                   {Compacted}
        shl (32|M0)              r6.0<1>:d     r4.0<1;1,0>:d     2:w               {I@1}
        send.slm (32|M0)         null     r6  r40:2  0x180000            0x04000504           {I@1,$14} // wr:2+2, rd:0; store.slm.d32.a32.flat[A+0x180]
        send.slm (32|M0)         null     r6  r32:2  0x80000            0x04000504           {F@1,$15} // wr:2+2, rd:0; store.slm.d32.a32.flat[A+0x80]
L7680:
        join (32|M0)                         L7696                                
L7696:
(W)     send.slm (1|M0)          r4       r38  null:0  0x0            0x0210001F           {$0} // wr:1+0, rd:1; fence.slm.none.group
(W)     mov (8|M0)               null<1>:ud    r4.0<1;1,0>:ud                   {$0.dst}
(W)     send.ugm (1|M0)          r5       r38  null:0  0x0            0x0210001F           {$1} // wr:1+0, rd:1; fence.ugm.none.group
(W)     mov (1|M0)               r6.2<1>:f     3.58732e-43:f                               {$15.src}
(W)     mov (2|M0)               r6.10<1>:ub   r38.11<0;1,0>:ub                 {F@1}
(W)     mov (8|M0)               null<1>:ud    r5.0<1;1,0>:ud                   {$1.dst}
(W)     send.gtwy (1|M0)         null     r6  null:0  0x0            0x02000004           {I@2,$2} // wr:1+0, rd:0; signal barrier
(W)     sync.bar                             0x0                                
(W)     mov (1|M0)               f3.0<1>:ud    r39.8<0;1,0>:ud                 
(~f3.0) goto (32|M0)                         L9472                  L9472                
L7856:
(W)     shr (1|M0)               r2.8<1>:ud    r3.6<0;1,0>:ud    5:w              
        cmp (32|M0)   (gt)f3.0   null<1>:ud    r2.8<0;1,0>:ud    r30.0<1;1,0>:ud  {I@1}
(f3.0)  goto (32|M0)                         L7936                  L7936                
L7904:
        mov (32|M0)              r52.0<1>:f    -inf:f                               {Compacted}
        mov (32|M0)              r32.0<1>:d    r2.6<0;1,0>:d                    {Compacted}
        goto (32|M0)                         L7936                  L8000                
L7936:
        join (32|M0)                         L8000                                
L7952:
        shl (32|M0)              r4.0<1>:d     r22.0<1;1,0>:d    2:w              
        send.slm (32|M0)         r52      r4  null:0  0x180000            0x04200500           {A@1,$3} // wr:2+0, rd:2; load.slm.d32.a32.flat[A+0x180]
        send.slm (32|M0)         r32      r4  null:0  0x80000            0x04200500           {$4} // wr:2+0, rd:2; load.slm.d32.a32.flat[A+0x80]
L8000:
        join (32|M0)                         L9472                                
L8016:
        sync.nop                             null                             {Compacted,$4.src}
(W)     mov (32|M0)              r4.0<1>:f     r52.0<1;1,0>:f                   {$3.dst}
(W)     mov (32|M0)              r6.0<1>:d     0:w                               {$2.src}
(W)     mov (1|M0)               f3.0<1>:ud    r39.3<0;1,0>:ud                  {Compacted}
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {$4.dst}
        mov (16|M0)              r8.0<1>:d     r5.0<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r6.0<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r13.0<1>:d    r6.0<1;1,0>:d                    {Compacted}
        mov (16|M0)              r12.0<1>:d    r11.0<1;1,0>:d                   {Compacted,F@1}
(f3.0)  sel (32|M0)              r84.0<1>:f    r8.0<1;1,0>:f     r52.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r4.0<1>:d     r12.0<1;1,0>:d    r32.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r84.0<1;1,0>:f    r52.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L8256                  L8256                
L8176:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r4.0<1;1,0>:d     r32.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r84.0<1;1,0>:f    r52.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L8256                  L8256                
L8224:
        mov (32|M0)              r60.0<1>:ud   r52.0<1;1,0>:ud                 
        goto (32|M0)                         L8256                  L8304                
L8256:
        join (32|M0)                         L8304                                
L8272:
        mov (32|M0)              r52.0<1>:f    r84.0<1;1,0>:f                   {Compacted,I@3}
        mov (32|M0)              r32.0<1>:d    r4.0<1;1,0>:d                    {Compacted}
        mov (32|M0)              r60.0<1>:f    r84.0<1;1,0>:f                  
L8304:
        join (32|M0)                         L9472                                
L8320:
(W)     mov (32|M0)              r4.0<1>:f     r52.0<1;1,0>:f                   {A@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (1|M0)               f3.0<1>:ud    r39.2<0;1,0>:ud                 
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                  
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.8<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.8<1;1,0>:d                    {Compacted,I@4}
        mov (16|M0)              r14.0<1>:d    r10.8<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.8<1;1,0>:d                   {Compacted,I@4}
(f3.0)  sel (32|M0)              r82.0<1>:f    r8.0<1;1,0>:f     r60.0<1;1,0>:f   {I@3}
(f3.0)  sel (32|M0)              r126.0<1>:d   r14.0<1;1,0>:d    r32.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f3.0   null<1>:f     r82.0<1;1,0>:f    r60.0<1;1,0>:f   {A@1}
(f3.0)  goto (32|M0)                         L8576                  L8576                
L8496:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r126.0<1;1,0>:d   r32.0<1;1,0>:d   {F@1}
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r82.0<1;1,0>:f    r60.0<1;1,0>:f   {I@1}
(f3.0)  goto (32|M0)                         L8576                  L8576                
L8544:
        mov (32|M0)              r58.0<1>:ud   r60.0<1;1,0>:ud                 
        goto (32|M0)                         L8576                  L8624                
L8576:
        join (32|M0)                         L8624                                
L8592:
        mov (32|M0)              r52.0<1>:f    r82.0<1;1,0>:f                   {Compacted}
        mov (32|M0)              r32.0<1>:d    r126.0<1;1,0>:d                  {Compacted}
        mov (32|M0)              r58.0<1>:f    r82.0<1;1,0>:f                   {I@4}
L8624:
        join (32|M0)                         L9472                                
L8640:
(W)     mov (32|M0)              r4.0<1>:f     r52.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {I@3}
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.4<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.4<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r14.0<1>:d    r10.4<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.4<1;1,0>:d                   {Compacted,I@4}
(f2.0)  sel (32|M0)              r80.0<1>:f    r8.0<1;1,0>:f     r58.0<1;1,0>:f   {I@3}
(f2.0)  sel (32|M0)              r124.0<1>:d   r14.0<1;1,0>:d    r32.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f2.0   null<1>:f     r80.0<1;1,0>:f    r58.0<1;1,0>:f   {A@1}
(f2.0)  goto (32|M0)                         L8872                  L8872                
L8800:
        cmp (32|M0)   (lt)f2.0   null<1>:d     r124.0<1;1,0>:d   r32.0<1;1,0>:d   {F@1}
(f2.0)  cmp (32|M0)   (eq)f2.0   null<1>:f     r80.0<1;1,0>:f    r58.0<1;1,0>:f   {I@1}
(f2.0)  goto (32|M0)                         L8872                  L8872                
L8848:
        mov (32|M0)              r56.0<1>:ud   r58.0<1;1,0>:ud                  {Compacted}
        goto (32|M0)                         L8872                  L8912                
L8872:
        join (32|M0)                         L8912                                
L8888:
        mov (32|M0)              r52.0<1>:f    r80.0<1;1,0>:f                   {Compacted}
        mov (32|M0)              r32.0<1>:d    r124.0<1;1,0>:d                  {Compacted}
        mov (32|M0)              r56.0<1>:f    r80.0<1;1,0>:f                   {Compacted,I@4}
L8912:
        join (32|M0)                         L9472                                
L8928:
(W)     mov (32|M0)              r4.0<1>:f     r52.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {I@3}
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.2<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.2<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r14.0<1>:d    r10.2<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.2<1;1,0>:d                   {Compacted,I@4}
(f1.0)  sel (32|M0)              r78.0<1>:f    r8.0<1;1,0>:f     r56.0<1;1,0>:f   {I@3}
(f1.0)  sel (32|M0)              r122.0<1>:d   r14.0<1;1,0>:d    r32.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f1.0   null<1>:f     r78.0<1;1,0>:f    r56.0<1;1,0>:f   {A@1}
(f1.0)  goto (32|M0)                         L9168                  L9168                
L9088:
        cmp (32|M0)   (lt)f2.0   null<1>:d     r122.0<1;1,0>:d   r32.0<1;1,0>:d  
(f2.0)  cmp (32|M0)   (eq)f2.0   null<1>:f     r78.0<1;1,0>:f    r56.0<1;1,0>:f   {I@1}
(f2.0)  goto (32|M0)                         L9168                  L9168                
L9136:
        mov (32|M0)              r94.0<1>:ud   r56.0<1;1,0>:ud                 
        goto (32|M0)                         L9168                  L9216                
L9168:
        join (32|M0)                         L9216                                
L9184:
        mov (32|M0)              r52.0<1>:f    r78.0<1;1,0>:f                   {Compacted}
        mov (32|M0)              r32.0<1>:d    r122.0<1;1,0>:d                  {Compacted}
        mov (32|M0)              r94.0<1>:f    r78.0<1;1,0>:f                   {I@4}
L9216:
        join (32|M0)                         L9472                                
L9232:
(W)     mov (32|M0)              r4.0<1>:f     r52.0<1;1,0>:f                   {F@2}
(W)     mov (32|M0)              r6.0<1>:d     0:w                              
(W)     mov (32|M0)              r10.0<1>:f    r32.0<1;1,0>:f                   {I@3}
(W)     mov (32|M0)              r12.0<1>:d    0:w                              
        mov (16|M0)              r8.0<1>:d     r4.1<1;1,0>:d                    {Compacted,F@2}
        mov (16|M0)              r9.0<1>:d     r5.1<1;1,0>:d                    {Compacted,I@3}
        mov (16|M0)              r14.0<1>:d    r10.1<1;1,0>:d                   {Compacted,F@1}
        mov (16|M0)              r15.0<1>:d    r11.1<1;1,0>:d                   {Compacted,I@4}
(f0.0)  sel (32|M0)              r6.0<1>:f     r94.0<1;1,0>:f    r8.0<1;1,0>:f    {I@3}
(f0.0)  sel (32|M0)              r120.0<1>:d   r32.0<1;1,0>:d    r14.0<1;1,0>:d   {I@1}
        cmp (32|M0)   (gt)f0.0   null<1>:f     r6.0<1;1,0>:f     r94.0<1;1,0>:f   {A@1}
(f0.0)  goto (32|M0)                         L9440                  L9440                
L9392:
        cmp (32|M0)   (lt)f3.0   null<1>:d     r120.0<1;1,0>:d   r32.0<1;1,0>:d  
(f3.0)  cmp (32|M0)   (eq)f3.0   null<1>:f     r6.0<1;1,0>:f     r94.0<1;1,0>:f   {I@1}
(~f3.0) goto (32|M0)                         L9440                  L9472                
L9440:
        join (32|M0)                         L9472                                
L9456:
        mov (32|M0)              r32.0<1>:f    r120.0<1;1,0>:f                 
L9472:
        join (32|M0)                         L9488                                
L9488:
        cmp (32|M0)   (ne)f3.0   null<1>:d     r28.0<1;1,0>:d    0:w               {F@2}
(~f3.0) goto (32|M0)                         L9688                  L9688                
L9520:
        cmp (32|M0)   (lt)f2.0   null<1>:d     r32.0<1;1,0>:d    r2.6<0;1,0>:d    {F@1}
(W)     shl (1|M0)               r2.4<1>:q     r38.6<0;1,0>:ud   2:w              
(f2.0)  sel (32|M0)              r4.0<1>:d     r32.0<1;1,0>:d    0:w               {$4.src}
(W)     add (1|M0)               r6.0<1>:q     r2.4<0;1,0>:q     r2.2<0;1,0>:q    {Compacted,@2,$2.src}
(W)     mov (1|M0)               r11.0<1>:uq   r39.0<0;1,0>:uq                  {Compacted}
        mov (16|M0)              r7.0<2>:f     r6.0<0;1,0>:f                    {I@2}
        mov (16|M16)             r9.0<2>:f     r6.0<0;1,0>:f                   
        mov (16|M0)              r7.1<2>:f     r6.1<0;1,0>:f                   
        mov (16|M16)             r9.1<2>:ud    r6.1<0;1,0>:ud                   {F@2}
        send.ugm (32|M0)         null     r7  r4:2  0x0            0x08000584           {A@1,$5} // wr:4+2, rd:0; store.ugm.d32.a64
(W)     mov (1|M0)               r12.0<1>:d    0:w                               {Compacted}
(W)     send.ugm (1|M0)          null     r11  r12:1  0x0            0x02008584           {I@1,$6} // wr:1+1, rd:0; store.ugm.d32x1t.a64
L9688:
        join (32|M0)                         L9704                                
L9704:
(W)     mov (16|M0)              r127.0<1>:f   r38.0<1;1,0>:f                   {Compacted}
(W)     send.gtwy (1|M0)         null     r127  null:0  0x0            0x02000010           {EOT,F@1,$7} // wr:1+0, rd:0; end of thread
L9728:
(W)     mov (16|M0)              null<1>:ud    0xEDC6AD7B:ud                             
(W)     mov (16|M0)              null<1>:ud    0x6C5F5BE2:ud                             
(W)     mov (16|M0)              null<1>:ud    0x0:ud                             
(W)     mov (16|M0)              null<1>:ud    0x1:ud                             
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
        illegal                
