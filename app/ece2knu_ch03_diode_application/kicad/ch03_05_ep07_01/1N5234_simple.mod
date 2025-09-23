* 1N5234 Zener diode - simple ngspice model
.model 1N5234 D( IS=1.5n  RS=0.5   N=1.5  XTI=3  EG=1.11
+                CJO=180p VJ=0.75  M=0.33 FC=0.5
+                BV=6.2   IBV=5m )
