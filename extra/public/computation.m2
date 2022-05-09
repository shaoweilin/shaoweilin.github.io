-- Shaowei Lin 16 Apr 2012
-- University of Washington

--------------------------------
-- EXAMPLE: Coin Toss Example --
--------------------------------

restart
load "~/Documents/Research/01 Bayesian Integrals/Programs/LIB Asymptotics/asymptotics.m2";

R = QQ[x,y];
I = ideal(x*y);
RLCT(I,1)


------------------------------
-- EXAMPLE: Schizo Patients --
------------------------------

restart
load "~/Documents/Research/01 Bayesian Integrals/Programs/LIB Asymptotics/asymptotics.m2";

-- define model
R = QQ[t,a1,a2,b1,b2,c1,c2,d1,d2];
A = matrix {{a1,a2,1-a1-a2}};
B = matrix {{b1,b2,1-b1-b2}};
C = matrix {{c1,c2,1-c1-c2}};
D = matrix {{d1,d2,1-d1-d2}};
P = t*(transpose A)*B + (1-t)*(transpose C)*D

-- translation and evaluation maps
shift = map(R,R,{t+1/2,a1+1/3,a2+1/3,b1+1/3,b2+1/3,c1+1/3,c2+1/3,d1+1/3,d2+1/3});
eval = map(R,R,{1/2,1/3,1/3,1/3,1/3,1/3,1/3,1/3,1/3});

-- fiber ideal
eval P
I = ideal (shift P - eval P)
I = ideal gens gb I

-- upper bound of RLCT
RLCT(I,1)

----------------------------------------------------------------------
-- One of the generators preventing I from being a monomial ideal is
--     "2t*b2 - 2t*d2 + b2 + d2".
-- We hope to replace it with a new variable "bb2" and to remove "b2".
-- Indeed, we can do this using the change of variable 
--           bb2 - (1-2t)*d2
--     b2 = -----------------
--                1+2t
-- which is well-defined near the origin.
--
-- In ALGEBRAIC GEOMETRY, we can also accomplish this by introducing 
-- the relation "-bb2 + 2t*b2 - 2t*d2 + b2 + d2" and eliminating 
-- the old variable "b2" from the ideal.
----------------------------------------------------------------------

R1 = QQ[t,a1,a2,b1,b2,c1,c2,d1,d2,bb1,bb2,cc1,cc2];
liftR1 = map(R1,R,{t,a1,a2,b1,b2,c1,c2,d1,d2});
I1 = (liftR1 I) + ideal(
     -bb2 + 2*t*b2 - 2*t*d2 + b2 + d2, 
     -bb1 + 2*t*b1 - 2*t*d1 + b1 + d1,
     -cc2 + 2*t*a2 - 2*t*c2 + a2 + c2,
     -cc1 + 2*t*a1 - 2*t*c1 + a1 + c1);
I1 = eliminate({c1,c2,b1,b2},I1)

-- RLCT of monomial ideal
RLCT(I1,1)