; -----------------------------------------------------------------------------
; High-Performance L2 Squared Distance for AMD64 (AVX2 + FMA)
; extern float l2_sq_dist_avx2(uint32_t dim, const float* a, const float* b);
; -----------------------------------------------------------------------------

section .text
global l2_sq_dist_avx2

l2_sq_dist_avx2:
    ; Parameters (Microsoft x64 Calling Convention / System V):
    ; RDI (edi) = dim
    ; RSI       = a
    ; RDX       = b
    
    vxorps ymm0, ymm0, ymm0 ; Clear accumulator ymm0 (8 floats)
    vxorps ymm1, ymm1, ymm1 ; Clear accumulator ymm1
    vxorps ymm2, ymm2, ymm2 ; Clear accumulator ymm2
    vxorps ymm3, ymm3, ymm3 ; Clear accumulator ymm3

    mov ecx, edi            ; dim
    shr ecx, 5              ; dim / 32 (unroll by 4 * 8-float lanes)
    jz .handle_remainder_8

.loop_32:
    ; Load 32 floats from A and B
    vmovups ymm4, [rsi]
    vmovups ymm5, [rsi + 32]
    vmovups ymm6, [rsi + 64]
    vmovups ymm7, [rsi + 96]
    
    vmovups ymm8, [rdx]
    vmovups ymm9, [rdx + 32]
    vmovups ymm10, [rdx + 64]
    vmovups ymm11, [rdx + 96]

    ; Calculate differences (a - b)
    vsubps ymm4, ymm4, ymm8
    vsubps ymm5, ymm5, ymm9
    vsubps ymm6, ymm6, ymm10
    vsubps ymm7, ymm7, ymm11

    ; Fused Multiply-Add: acc = acc + (diff * diff)
    vfmadd231ps ymm0, ymm4, ymm4
    vfmadd231ps ymm1, ymm5, ymm5
    vfmadd231ps ymm2, ymm6, ymm6
    vfmadd231ps ymm3, ymm7, ymm7

    add rsi, 128
    add rdx, 128
    dec ecx
    jnz .loop_32

.handle_remainder_8:
    mov ecx, edi
    and ecx, 31             ; Remainder after 32-wide chunks
    shr ecx, 3              ; How many 8-float chunks?
    jz .handle_scalar

.loop_8:
    vmovups ymm4, [rsi]
    vmovups ymm8, [rdx]
    vsubps ymm4, ymm4, ymm8
    vfmadd231ps ymm0, ymm4, ymm4
    add rsi, 32
    add rdx, 32
    dec ecx
    jnz .loop_8

.handle_scalar:
    ; Sum the four accumulators into ymm0
    vaddps ymm0, ymm0, ymm1
    vaddps ymm0, ymm0, ymm2
    vaddps ymm0, ymm0, ymm3

    ; Horizontal sum of ymm0
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    ; Handle final 1-7 floats (scalar)
    mov ecx, edi
    and ecx, 7
    jz .done

.scalar_loop:
    vmovss xmm1, [rsi]
    vsubss xmm1, xmm1, [rdx]
    vmulss xmm1, xmm1, xmm1
    vaddss xmm0, xmm0, xmm1
    add rsi, 4
    add rdx, 4
    dec ecx
    jnz .scalar_loop

.done:
    vzeroupper
    ret
