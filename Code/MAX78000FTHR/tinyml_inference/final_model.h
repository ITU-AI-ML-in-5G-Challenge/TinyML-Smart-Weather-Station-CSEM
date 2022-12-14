/*
; final_model.h.
; =========

;------------------------------------------------------------------------
; Author:	Robin Berguerand, Jona Beysens, Jonathan Reymond	The 2022-12-14
; Modifs:
;
; Goal:	Tensorflow lite micro model header
;
;   (c) 1992-2022, Edo. Franzi
;   --------------------------
;
;   CSEM S.A.
;   Jaquet-Droz 1
;   CH-2000 Neuchâtel
;   http://www.csem.ch
;
;   ____________________/\\\______/\\\______/\\\_
;    ________________/\\\\\\\__/\\\\\\\__/\\\\\\\_
;     _______________\/////\\\_\/////\\\_\/////\\\_
;      __/\\\____/\\\_____\/\\\_____\/\\\_____\/\\\_
;       _\/\\\___\/\\\_____\/\\\_____\/\\\_____\/\\\_
;        _\//\\\\\\\\\______\/\\\_____\/\\\_____\/\\\_
;         __\/////////_______\///______\///______\///_
;
;   THIS PROGRAM IS CONFIDENTIAL AND CANNOT BE DISTRIBUTED
;   WITHOUT THE CSEM PRIOR WRITTEN AGREEMENT.
;
;   u111 is an optimised branch of uKOS-III package.
;   CSEM is the owner of this branch and is authorised to use, to modify
;   and to keep confidential all new adaptations of this branch.
;------------------------------------------------------------------------
*/

#include <cstdint>

extern const unsigned char final_model[];
extern const unsigned int final_model_len;
