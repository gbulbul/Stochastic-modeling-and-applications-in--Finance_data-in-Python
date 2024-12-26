# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 10:55:27 2024

@author: gbulb
"""

import	numpy	as	np
class Monte_carlo_sim:
    def val_of_european_call(S0,K,T,r,sigma,I,z):
        ST	=	S0	*	np.exp((r	-	0.5	*	sigma	**	2)	*	T	+	sigma	*	np.sqrt(T)	*	z)
        hT	=	np.maximum(ST	-	K,	0)		
        C0	=	np.exp(-r	*	T)	*	np.sum(hT)	/	I		
        return C0
    


if __name__=="__main__":
    S0	=	100		#	initial	index	level
    K	=	105		#	strike	price
    T	=	1.0		#	time-to-maturity
    r	=	0.05		#	riskless	short	rate
    sigma	=	0.2		#	volatility
    I	=	100000		#	number	of	simulations
    z	=	np.random.standard_normal(I)		#	pseudorandom	numbers
    C0=Monte_carlo_sim.val_of_european_call(S0,K,T,r,sigma,I,z)
    print	("Value	of	the	European	Call	Option	%5.3f"	%	C0)
