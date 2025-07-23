"""
All functions support the following formats: GRAY and YUV 8 - 16 bit integer.

Support for floating point sample type is added when needed. Such functions are marked separately.
Support for floating point sample type is intended for clips converted from the full range.
For clips converted from a limited range, correct operation is not guaranteed.

Functions:
    autotap3 (float support)
    Lanczosplus
    bion_dehalo (float support)
    fix_border (float support)
    MaskDetail (float only)
    degrain_n
    Destripe (float only)
    daa
    average_fields
    znedi3aas
    dehalo_mask
    tp7_deband_mask
    DeHalo_alpha
    FineDehalo
    FineDehalo2
    upscaler (float support)
    diff_mask
    apply_range (float support)
    titles_mask
    after_mask (float support)
    search_field_diffs
    CombMask2
    MTCombMask
    Binarize
    delcomb
    vinverse
    vinverse2
    sbr (float support)
    sbrV (float support)
    Blur (float support)
    Sharpen (float support)
    Clamp (float support)
    MinBlur (float support)
    DitherLumaRebuild
    ExpandMulti
    InpandMulti
    TemporalSoften
    UnsharpMask
    diff_tfm
    diff_transfer
    shift_clip (float support)
    ovr_comparator
    RemoveGrain (float support)
    Repair (float support)
    TemporalRepair (float support)
    Clense (float support)
    BackwardClense (float support)
    ForwardClense (float support)
    VerticalCleaner (float support)
    Convolution (float support)
    CrazyPlaneStats (float support)
    out_of_range_search
    rescaler (float only)
    SCDetect
    getnative (float only)
"""

import re
from collections.abc import Callable
from functools import partial, wraps
from inspect import signature
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Self

import numpy as np
import vapoursynth as vs
from vapoursynth import core


def float_decorator(num_clips: int = 1) -> Callable:
    
    func_name = 'float_decorator'
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
            
            if not all(isinstance(i, vs.VideoNode) for i in args[:num_clips]):
                raise TypeError(f'{func_name} the clip(s) must be of the vs.VideoNode type')
            
            if not all(i.format.color_family in {vs.YUV, vs.GRAY} for i in args[:num_clips]):
                raise TypeError(f'{func_name}: Unsupported color family')
            
            if all(i.format.sample_type == vs.INTEGER for i in args[:num_clips]):
                return func(*args, **kwargs)
            elif all(i.format.sample_type == vs.FLOAT for i in args[:num_clips]):
                pass
            else:
                raise TypeError(f'{func_name} the clip(s) must be of the INTEGER or FLOAT sample type')
            
            expr0 = ['x 1 min 0 max', 'x 0.5 + 1 min 0 max', 'x 0.5 + 1 min 0 max']
            expr1 = ['x 1 min 0 max', 'x 0.5 - 0.5 min -0.5 max', 'x 0.5 - 0.5 min -0.5 max']
            
            clips = [core.std.Expr(i, expr0[:i.format.num_planes]) for i in args[:num_clips]]
            clip = func(*clips, *args[num_clips:], **kwargs)
            clip = core.std.Expr(clip, expr1[:clip.format.num_planes])
            
            return clip
        
        return wrapper
    
    return decorator

def chroma_up(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 1 min 0 max', 'x 0.5 + 1 min 0 max', 'x 0.5 + 1 min 0 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def chroma_down(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 1 min 0 max', 'x 0.5 - 0.5 min -0.5 max', 'x 0.5 - 0.5 min -0.5 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def luma_up(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 0.5 + 1 min 0 max', 'x 0.5 min -0.5 max', 'x 0.5 min -0.5 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def luma_down(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 0.5 - 0.5 min -0.5 max', 'x 0.5 min -0.5 max', 'x 0.5 min -0.5 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def diff_clamp(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 0.5 min -0.5 max', 'x 0.5 min -0.5 max', 'x 0.5 min -0.5 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def clip_clamp(clip: vs.VideoNode, planes: list[int]) -> vs.VideoNode:
    
    if clip.format.sample_type == vs.FLOAT:
        expr = ['x 1 min 0 max', 'x 0.5 min -0.5 max', 'x 0.5 min -0.5 max']
        clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(clip.format.num_planes)])
    
    return clip

def autotap3(clip: vs.VideoNode, dx: int | None = None, dy: int | None = None, mtaps3: int = 1, thresh: int = 256,
             **crop_args: float) -> vs.VideoNode:
    """
    Lanczos-based resize from "*.mp4 guy", ported from AviSynth version with minor modifications.
    
    In comparison with the original, processing accuracy has been doubled, support for 8-32 bit depth
    and crop parameters has been added, and dead code has been removed.
    
    dx and dy are the desired resolution.
    The other parameters are not documented in any way and are selectedusing the poke method.
    Cropping options are added as **kwargs. The key names are the same as in VapourSynth-resize.
    """
    func_name = 'autotap3'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    match dx:
        case None:
            dx = w * 2
        case int():
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "dx"')
    
    match dy:
        case None:
            dy = h * 2
        case int():
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "dy"')
    
    if not isinstance(mtaps3, int) or mtaps3 <= 0 or mtaps3 > 128:
        raise TypeError(f'{func_name}: invalid "mtaps3"')
    
    if not isinstance(thresh, int) or thresh <= 0 or thresh > 256:
        raise TypeError(f'{func_name}: invalid "thresh"')
    
    back_args = {}
    
    if crop_args:
        if 'src_left' in crop_args:
            back_args['src_left'] = -crop_args['src_left'] * dx / w
        
        if 'src_top' in crop_args:
            back_args['src_top'] = -crop_args['src_top'] * dy / h
        
        if 'src_width' in crop_args:
            if crop_args['src_width'] <= 0:
                crop_args['src_width'] += w - crop_args.get('src_left', 0)
            back_args['src_width'] = dx * 2 - crop_args['src_width'] * dx / w
        
        if 'src_height' in crop_args:
            if crop_args['src_height'] <= 0:
                crop_args['src_height'] += h - crop_args.get('src_top', 0)
            back_args['src_height'] = dy * 2 - crop_args['src_height'] * dy / h
        
        if any((x := i) not in back_args for i in crop_args):
            raise KeyError(f'{func_name}: Unsupported key {x} in crop_args')
    
    if dx == w and dy == h:
        return core.resize.Spline36(clip, **crop_args)
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    t1 = core.resize.Lanczos(clip, dx, dy, filter_param_a=1, **crop_args)
    t2 = core.resize.Lanczos(clip, dx, dy, filter_param_a=2, **crop_args)
    t3 = core.resize.Lanczos(clip, dx, dy, filter_param_a=3, **crop_args)
    t4 = core.resize.Lanczos(clip, dx, dy, filter_param_a=4, **crop_args)
    t5 = core.resize.Lanczos(clip, dx, dy, filter_param_a=5, **crop_args)
    t6 = core.resize.Lanczos(clip, dx, dy, filter_param_a=9, **crop_args)
    t7 = core.resize.Lanczos(clip, dx, dy, filter_param_a=36, **crop_args)
    
    m1 = core.std.Expr([clip, core.resize.Lanczos(t1, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m2 = core.std.Expr([clip, core.resize.Lanczos(t2, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m3 = core.std.Expr([clip, core.resize.Lanczos(t3, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m4 = core.std.Expr([clip, core.resize.Lanczos(t4, w, h, filter_param_a=2, **back_args)], 'x y - abs')
    m5 = core.std.Expr([clip, core.resize.Lanczos(t5, w, h, filter_param_a=2, **back_args)], 'x y - abs')
    m6 = core.std.Expr([clip, core.resize.Lanczos(t6, w, h, filter_param_a=3, **back_args)], 'x y - abs')
    m7 = core.std.Expr([clip, core.resize.Lanczos(t7, w, h, filter_param_a=6, **back_args)], 'x y - abs')
    
    expr = f'x y - {thresh} *' if clip.format.sample_type == vs.INTEGER else f'x y - {thresh} * 1 min 0 max'
    
    cp0 = Blur(t1, 1.42)
    m100 = core.std.Expr([m1, m2], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    cp1 = core.std.MaskedMerge(cp0, t2, m100)
    m101 = core.std.Expr([clip, core.resize.Bilinear(cp1, w, h, **back_args)], 'x y - abs')
    m101 = core.std.Expr([m101, m3], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    cp2 = core.std.MaskedMerge(cp1, t3, m101)
    m102 = core.std.Expr([clip, core.resize.Bilinear(cp2, w, h, **back_args)], 'x y - abs')
    m102 = core.std.Expr([m102, m4], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    cp3 = core.std.MaskedMerge(cp2, t4, m102)
    m103 = core.std.Expr([clip, core.resize.Bilinear(cp3, w, h, **back_args)], 'x y - abs')
    m103 = core.std.Expr([m103, m5], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    cp4 = core.std.MaskedMerge(cp3, t5, m103)
    m104 = core.std.Expr([clip, core.resize.Bilinear(cp4, w, h, **back_args)], 'x y - abs')
    m104 = core.std.Expr([m104, m6], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    cp5 = core.std.MaskedMerge(cp4, t6, m104)
    m105 = core.std.Expr([clip, core.resize.Bilinear(cp5, w, h, **back_args)], 'x y - abs')
    m105 = core.std.Expr([m105, m7], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args)
    clip = core.std.MaskedMerge(cp5, t7, m105)
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy, **crop_args)],
                                      list(range(orig.format.num_planes)), space)
    
    return clip

def Lanczosplus(clip: vs.VideoNode, dx: int | None = None, dy: int | None = None, thresh: int = 0,
                thresh2: int | None = None, athresh: int = 256, sharp1: float = 1, sharp2: float = 4,
                blur1: float = 0.33, blur2: float = 1.25, mtaps1: int = 1, mtaps2: int = 1, ttaps: int = 1,
                ltaps: int = 1, preblur: bool = False, depth: int = 2, wthresh: int = 230, wblur: int = 2,
                mtaps3: int = 1) -> vs.VideoNode:
    """
    An upscaler based on Lanczos and AWarpSharp from "*.mp4 guy", ported from AviSynth version with minor modifications.
    
    In comparison with the original, the mathematics for non-multiple resolutions has been improved,
    support for 8-16 bit depth has been added, dead code and unnecessary calculations have been removed.
    All dependent parameters have been recalculated from AWarpSharp to AWarpSharp2.
    It comes with autotap3, ported just for completion.
    
    dx and dy are the desired resolution.
    The other parameters are not documented in any way and are selected using the poke method.
    """
    func_name = 'Lanczosplus'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w * 2
    
    if dy is None:
        dy = h * 2
    
    if dx <= w or dy <= h:
        raise ValueError(f'{func_name}: this is an upscaler, dx and dy must be larger than the width and height')
    
    if thresh2 is None:
        thresh2 = (thresh + 1) * 64
    
    space = clip.format.color_family
    bits = clip.format.bits_per_sample
    thresh *= 1 << bits - 8
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    fd1 = core.resize.Lanczos(clip, dx, dy, filter_param_a=mtaps1)
    fre1 = core.resize.Lanczos(fd1, w, h, filter_param_a=mtaps1)
    fre2 = autotap3(fre1, x := max(w // 16 * 8, 144), y := max(h // 16 * 8, 144), mtaps3, athresh)
    fre2 = autotap3(fre2, w, h, mtaps3, athresh)
    m1 = core.std.Expr([fre1, clip], f'x y - abs {thresh} - {thresh2} *')
    m2 = (core.frfun7.Frfun7(m1, l=2.01, t=256, tuv=256, p=1) if bits == 8 else
          core.fmtc.bitdepth(m1, bits=8, dmode=1).frfun7.Frfun7(l=2.01, t=256, tuv=256, p=1).fmtc.bitdepth(bits=bits))
    m2 = core.resize.Lanczos(core.resize.Lanczos(m2, x, y, filter_param_a=ttaps), dx, dy, filter_param_a=ttaps)
    
    d = core.std.MaskedMerge(clip, fre2, m1) if preblur else clip
    d2 = autotap3(d, dx, dy, mtaps3, athresh)
    d3 = core.resize.Lanczos(core.resize.Lanczos(d, w, h, filter_param_a=ttaps), dx, dy, filter_param_a=ttaps)
    d4 = core.std.MaskedMerge(core.std.Expr([d2, d3],  f'x y - {sharp1} * x +'),
                              core.std.Expr([d2, d3],  f'y x - {blur1} * x +'), m2)
    d5 = autotap3(d4, w, h, mtaps3, athresh)
    
    e = autotap3(core.std.MaskedMerge(d5, clip, m1), dx, dy, mtaps3, athresh)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    
    fd12 = core.resize.Lanczos(e, dx ** 2 // w // 16 * 16, dy ** 2 // h // 16 * 16, filter_param_a=mtaps2)
    fre12 = core.resize.Lanczos(fd12, dx, dy, filter_param_a=mtaps2)
    m12 = core.std.Expr([fre12, e], f'x y - abs {thresh} - {thresh2} *')
    m12 = core.resize.Lanczos(m12, max(dx // 16 * 8, 144), max(dy // 16 * 8, 144), filter_param_a=mtaps2)
    m12 = core.resize.Lanczos(m12, dx, dy, filter_param_a=mtaps2)
    
    e2 = core.resize.Lanczos(core.resize.Lanczos(e, w, h, filter_param_a=ltaps), dx, dy, filter_param_a=ltaps)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    
    e3 = core.std.MaskedMerge(core.std.Expr([e, e2], f'y x - {blur2} * x +'),
                              core.std.Expr([e, e2], f'x y - {sharp2} * x +'), m12)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    
    clip = core.std.MaskedMerge(d4, e3, m2)
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy)],
                                      list(range(orig.format.num_planes)), space)
    
    return clip

def bion_dehalo(clip: vs.VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1,
                m: bool = False) -> vs.VideoNode:
    """
    Dehalo by bion, ported from AviSynth version with minor additions.
    
    Args:
        mode: Repair mode from dehaloed clip.
            1, 5, 11 - the weakest, artifacts will not cause.
            2, 3, 4 - bad modes, eat innocent parts, can't be used.
            10 - almost like mode = 1, 5, 11, but with a spread around the edges.
            I think it's a little better for noisy sources.
            14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.
        rep: use Repair to clamp result clip or not.
        rg: use RemoveGrain and Repair to merge with blurred clip or not.
        mask: the mask to merge clip and blurred clip.
            3 - the most accurate.
            4 - the roughest.
            1 and 2 - somewhere in the middle.
            5...7 - the same, but Gaussian convolution is used
        m: show the mask instead of the clip or not.
    """
    func_name = 'bion_dehalo'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if clip.format.sample_type == vs.INTEGER:
        factor = 1 << clip.format.bits_per_sample - 8
        half = 128 * factor
        expr0 = f'x {4 * factor} - 4 *'
        expr1 = 'x y 1.2 * -'
        expr2 = f'x y - {factor} - 128 *'
        expr3 = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs 2 * < x y {half} - 2 * {half} + ? ?'
    else:
        expr0 = 'x 4 255 / - 4 * 1 min 0 max'
        expr1 = 'x y 1.2 * - 1 min 0 max'
        expr2 = 'x y - 1 255 / - 128 * 1 min 0 max'
        expr3 = 'x y * 0 < 0 x abs y abs 2 * < x y 2 * ? ? 0.5 min -0.5 max'
    
    def get_mask(clip: vs.VideoNode, mask: int) -> vs.VideoNode:
        
        match mask:
            case 0:
                m1 = core.std.Expr(Convolution(clip, 'min/max'), expr0)
                m2 = core.std.Maximum(m1).std.Maximum()
                m2 = core.std.Merge(m2, core.std.Maximum(m2)).std.Inflate()
                m3 = core.std.Expr([core.std.Merge(m2, core.std.Maximum(m2)), core.std.Deflate(m1)], expr1)
                m3 = core.std.Inflate(m3)
            case 1:
                m1 = core.std.Expr([clip, UnsharpMask(clip, 40, 2, 0)], expr2).std.Maximum().std.Inflate()
                m2 = core.std.Maximum(m1).std.Maximum()
                m3 = RemoveGrain(core.std.Expr([m1, m2], 'y x -'), 21).std.Maximum()
            case 2:
                m1 = core.std.Expr([clip, UnsharpMask(clip, 40, 2, 0, 'gauss')], expr2).std.Maximum().std.Inflate()
                m2 = core.std.Maximum(m1).std.Maximum()
                m3 = RemoveGrain(core.std.Expr([m1, m2], 'y x -'), 21).std.Maximum()
        
        return m3
    
    match mask:
        case 1:
            mask = get_mask(clip, 0)
        case 2:
            mask = get_mask(clip, 1)
        case 3:
            mask = core.std.Expr([get_mask(clip, 0), get_mask(clip, 1)], 'x y min')
        case 4:
            mask = core.std.Expr([get_mask(clip, 0), get_mask(clip, 1)], 'x y max')
        case 5:
            mask = get_mask(clip, 2)
        case 6:
            mask = core.std.Expr([get_mask(clip, 0), get_mask(clip, 2)], 'x y min')
        case 7:
            mask = core.std.Expr([get_mask(clip, 0), get_mask(clip, 2)], 'x y max')
        case _:
            raise ValueError(f'{func_name}: Please use 1...7 mask value')
    
    blurr = RemoveGrain(RemoveGrain(MinBlur(clip, 1), 11), 11)
    
    if rg:
        dh1 = core.std.MaskedMerge(Repair(clip, RemoveGrain(clip, 21), 1), blurr, mask)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, mask)
    
    dh1D = diff_clamp(core.std.MakeDiff(clip, dh1), [0])
    tmp = sbr(dh1)
    med2D = diff_clamp(core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2)), [0])
    DD  = core.std.Expr([dh1D, med2D], expr3)
    dh2 = clip_clamp(core.std.MergeDiff(dh1, DD), [0])
    
    clip = Clamp(clip, Repair(clip, dh2, mode) if rep else dh2, clip, 0, 20)
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if m:
        clip = core.resize.Point(mask, format=orig.format.id) if space == vs.YUV else mask
    
    return clip

@float_decorator()
def fix_border(clip: vs.VideoNode, /, *args: list[str | int | list[int] | bool]) -> vs.VideoNode:
    """
    A simple functions for fix brightness artifacts at the borders of the frame.
    
    All values are set as positional list arguments. The list have the following format:
    [axis, target, donor, limit, curve, shift, plane, mean, clamp]. The first three are mandatory.
    
    Args:
        axis: can take the values "X" or "Y" for columns and rows respectively.
        target: the target column/row, it is counted from the upper left edge of the screen. It could be a list.
            It can be negative.
        donor: the donor column/row. It could be a list. It can be negative.
        limit: by default 0, without restrictions, positive values prohibit the darkening of target rows/columns
            and limit the maximum lightening, negative values - on the contrary, it's set in 8-bit notation.
        curve: target correction curve. 0 - subtraction and addition, -1 and 1 - division and multiplication,
            -2 and 2 - logarithm and exponentiation, -3 and 3 - nth root and exponentiation, by default 1.
        shift: shift of the zero point of the curve relative to the beginning of the range,
            if curve < 0 - the shift is relative to the end of the range, by default 0.
        plane: by default 0.
        mean: CrazyPlaneStats mode, by default 0.
        clamp: clamp target between donor minimum and maximum, by default True.
    
    Example:
        clip = fix_border(clip, ['X', 0, 1, 50], ['X', -1, -2, 50], ['Y', 0, 1, 50], ['Y', -1, -2, 50])
    """
    func_name = 'fix_border'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space == vs.YUV:
        clips = core.std.SplitPlanes(clip)
    elif space == vs.GRAY:
        clips = [clip]
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    def correction(clip: vs.VideoNode, axis: str, target: int | list[int], donor: int | list[int], limit: int,
                   curve: int, shift: int, mean: int, clamp: bool) -> vs.VideoNode:
        
        if not isinstance(shift, int):
            raise ValueError(f'{func_name}: "shift" must be "int"')
        
        if clip.format.sample_type == vs.INTEGER:
            factor = 1 << clip.format.bits_per_sample - 8
            shift *= factor
            full = 256 * factor - 1
        else:
            shift /= 255
            full = 1
        
        match abs(curve):
            case 0:
                expr = f'x.donor_avg {shift} + x.target_avg {shift} + - x {shift} + +'
            case 1:
                expr = f'x.donor_avg {shift} + x.target_avg {shift} + / x {shift} + *'
            case 2:
                expr = f'x {shift} + x.donor_avg {shift} + log x.target_avg {shift} + log / pow'
            case 3:
                expr = f'x.donor_avg {shift} + 1 x.target_avg {shift} + / pow x {shift} + pow'
            case _:
                raise ValueError(f'{func_name}: Please use -3...3 curve value')
        
        if clamp:
            expr = f'{expr} x.minimum {shift} + x.maximum {shift} + clamp'
        
        if curve < 0:
            clip = core.std.InvertMask(clip)
            limit = -limit
        
        match target:
            case int():
                target = [target]
            case list() if all(isinstance(i, int) for i in target):
                pass
            case _:
                raise TypeError(f'{func_name}: "target" must be "int" or "list[int]"')
        
        match donor:
            case int():
                donor = [donor]
            case list() if all(isinstance(i, int) for i in donor):
                pass
            case _:
                raise TypeError(f'{func_name}: "donor" must be "int" or "list[int]"')
        
        orig = clip
        
        def stats_x(clip: vs.VideoNode, x: list[int], w: int, mean: int) -> vs.VideoNode:
            
            if all(0 <= i < w for i in x):
                pass
            elif all(-w <= i < 0 for i in x):
                x[:] = [i + w for i in x]
            else:
                raise ValueError(f'{func_name}: {x} is out of range')
            
            return CrazyPlaneStats(
                core.std.StackHorizontal([core.std.Crop(clip, i, w - i - 1, 0, 0) for i in x]),
                mean, norm=False
            )
        
        def stats_y(clip: vs.VideoNode, y: list[int], h: int, mean: int) -> vs.VideoNode:
            
            if all(0 <= i < h for i in y):
                pass
            elif all(-h <= i < 0 for i in y):
                y[:] = [i + h for i in y]
            else:
                raise ValueError(f'{func_name}: {y} is out of range')
            
            return CrazyPlaneStats(
                core.std.StackVertical([core.std.Crop(clip, 0, 0, i, h - i - 1) for i in y]),
                mean, norm=False
            )
        
        match axis:
            case 'X':
                w = clip.width
                clip = core.akarin.PropExpr(
                    [clip, stats_x(clip, target, w, mean), stats_x(clip, donor, w, mean)],
                    lambda: dict(target_avg=f'y.{means[mean]}', donor_avg=f'z.{means[mean]}',
                                 maximum='z.maximum', minimum='z.minimum')
                )
            case 'Y':
                h = clip.height
                clip = core.akarin.PropExpr(
                    [clip, stats_y(clip, target, h, mean), stats_y(clip, donor, h, mean)],
                    lambda: dict(target_avg=f'y.{means[mean]}', donor_avg=f'z.{means[mean]}',
                                 maximum='z.maximum', minimum='z.minimum')
                )
            case _:
                raise ValueError(f'{func_name}: invalid "axis"')
        
        expr = (f'{' '.join(f'{axis} {i} =' for i in target)} '
                f'{'or ' * (len(target) - 1)}{expr} {shift} - 0 {full} clamp x ?')
        clip = core.akarin.Expr(clip, expr)
        clip = core.std.RemoveFrameProps(clip, ['target_avg', 'donor_avg', 'maximum', 'minimum'])
        
        if limit > 0:
            clip = Clamp(clip, orig, orig, limit, 0)
        elif limit < 0:
            clip = Clamp(clip, orig, orig, 0, -limit)
        
        if curve < 0:
            clip = core.std.InvertMask(clip)
        
        return clip
    
    defaults = ['X', 0, 0, 0, 1, 0, 0, 0, True]
    
    means = ['arithmetic_mean', 'geometric_mean', 'arithmetic_geometric_mean', 'harmonic_mean', 'contraharmonic_mean',
             'root_mean_square', 'root_mean_cube', 'median']
    
    for i in args:
        if isinstance(i, list) and 3 <= len(i) <= 9:
            if len(i) < 9:
                i += defaults[len(i):]
        else:
            raise ValueError(f'{func_name}: *args must be a sequence of lists with 3 <= len(list) <= 9')
        
        if i[6] in set(range(num_p)):
            clips[i[6]] = correction(clips[i[6]], *i[:6], *i[7:])
        else:
            raise ValueError(f'{func_name}: invalid plane {i[6]}')
    
    clip = core.std.ShufflePlanes(clips, [0] * num_p, space) if space == vs.YUV else clips[0]
    
    return clip

def mask_detail(clip: vs.VideoNode, dx: float | None = None, dy: float | None = None, rg: int = 3, cutoff: int = 70,
                gain: float = 0.75, exp_n: int = 2, inf_n: int = 1, blur_more: bool = False, kernel: str = 'bilinear',
                mode: int = 1, **descale_args: Any) -> vs.VideoNode:
    """
    MaskDetail by "Tada no Snob", ported from AviSynth version with minor additions.
    
    It is based on the internal rescale function, therefore it supports fractional resolutions
    and automatic width calculation based on the original aspect ratio.
    """
    func_name = 'mask_detail'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.FLOAT:
        raise TypeError(f'{func_name}: integer sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    match kernel:
        case 'bicubic':
            resc = rescaler(clip, dx, dy, kernel, mode, **descale_args)
        case 'lanczos':
            resc = rescaler(clip, dx, dy, kernel, mode, **descale_args)
        case _:
            resc = rescaler(clip, dx, dy, kernel, mode)
    
    expr = 'x y - 0.5 + 0 1 clamp 16 * var! var@ 1.0 % val! var@ trunc 1 bitand 1 = 1 val@ - val@ ?'
    clip = core.akarin.Expr([clip, resc], expr)
    clip = RemoveGrain(clip, rg)
    clip = core.std.Expr(clip, f'x {cutoff} 255 / < 0 x {gain} 1 x + * * 1 min 0 max ?')
    
    for _ in range(exp_n):
        clip = core.std.Maximum(clip)
    
    for _ in range(inf_n):
        clip = core.std.Inflate(clip)
    
    if blur_more:
        clip = RemoveGrain(clip, 12)
    
    if space == vs.YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def degrain_n(clip: vs.VideoNode, /, *args: dict[str, Any], tr: int = 1, full_range: bool = False) -> vs.VideoNode:
    """
    Just an alias for mv.Degrain.
    
    The parameters of individual functions are set as dictionaries. Unloading takes place sequentially,
    separated by commas. If you do not set anything, the default settings of MVTools itself apply.
    Function dictionaries are set in order: Super, Analyze, Degrain, Recalculate.
    Recalculate is optional, but you can specify several of them (as many as you want).
    If you need to specify settings for only one function, the rest of the dictionaries are served empty.
    """
    func_name = 'degrain_n'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if tr > 6 or tr < 1:
        raise ValueError(f'{func_name}: 1 <= "tr" <= 6')
    
    if len(args) < 3:
        args += ({},) * (3 - len(args))
    
    if full_range:
        sup1 = DitherLumaRebuild(clip, s0 = 1).mv.Super(rfilter=4, **args[0])
        sup2 = core.mv.Super(clip, levels=1, **args[0])
    else:
        sup1 = core.mv.Super(clip, **args[0])
    
    vectors = [core.mv.Analyse(sup1, isb=j, delta=i, **args[1]) for i in range(1, tr + 1) for j in (True, False)]
    
    for i in args[3:]:
        vectors = [core.mv.Recalculate(sup1, j, **i) for j in vectors]
    
    clip = getattr(core.mv, f'Degrain{tr}')(clip, sup2 if full_range else sup1, *vectors, **args[2])
    
    return clip

def Destripe(clip: vs.VideoNode, dx: int | None = None, dy: int | None = None, kernel: str = 'bilinear',
             tff: bool = True, **descale_args: Any) -> vs.VideoNode:
    """
    Simplified Destripe from YomikoR without unnecessary frills.
    
    The internal Descale functions are unloaded as usual.
    The function values that differ for the upper and lower fields are indicated in the list.
    """
    func_name = 'Destripe'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.FLOAT:
        raise TypeError(f'{func_name}: integer sample type is not supported')
    
    if dx is None:
        dx = clip.width
    
    if dy is None:
        dy = clip.height // 2
    
    second_args = {}
    
    for key, value in descale_args.items():
        if isinstance(value, list):
            if len(value) == 2:
                second_args[key] = value[1]
                descale_args[key] = value[0]
            else:
                raise ValueError(f'{func_name}: {key} length must be 2')
        else:
            second_args[key] = value
    
    clip = core.std.SetFieldBased(clip, 0).std.SeparateFields(tff).std.SetFieldBased(0)
    fields = [clip[::2], clip[1::2]]
    
    fields[0] = getattr(core.descale, f'De{kernel}')(fields[0], dx, dy, **descale_args)
    fields[1] = getattr(core.descale, f'De{kernel}')(fields[1], dx, dy, **second_args)
    
    clip = core.std.Interleave(fields)
    clip = core.std.DoubleWeave(clip, tff)[::2]
    clip = core.std.SetFieldBased(clip, 0)
    
    return clip

def daa(clip: vs.VideoNode, weight: float = 0.5, planes: int | list[int] | None = None,
        **znedi3_args: Any) -> vs.VideoNode:
    """daa by Didée, ported from AviSynth version with minor additions."""
    func_name = 'daa'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(weight, float | int) or weight < 0 or weight > 1:
        raise ValueError(f'{func_name}: invalid "weight"')
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    nn = core.znedi3.nnedi3(clip, field=3, planes=planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [weight if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes=planes)
    mode = 20 if clip.width > 1100 else 11
    shrpD = core.std.MakeDiff(dbl, RemoveGrain(dbl, [mode if i in planes else 0 for i in range(num_p)]), planes=planes)
    DD = Repair(shrpD, dblD, [13 if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes=planes)
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else dblD for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def average_fields(clip: vs.VideoNode, curve: int | list[int | None] = 1, weight: float = 0.5,
                   shift: int | list[int] = 0, mode: int = 0, mean: int = 0) -> vs.VideoNode:
    """
    Just an experiment. It leads to a common denominator of the average normalized values of the fields of one frame.
    
    Ideally, it should fix interlaced fades painlessly, but in practice this does not always happen.
    Apparently it depends on the source.
    """
    func_name = 'average_fields'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor - 1
    
    def simple_average(clip: vs.VideoNode, curve: int | None, weight: float, shift: int, mode: int,
                       mean: int) -> vs.VideoNode:
        
        if curve is None:
            return clip
        
        if weight == 0:
            expr0 = f'x.{means[mean]} {shift} +'
        elif weight == 1:
            expr0 = f'y.{means[mean]} {shift} +'
        elif 0 < weight < 1:
            expr0 = f'x.{means[mean]} {1 - weight} * y.{means[mean]} {weight} * + {shift} +'
        else:
            raise ValueError(f'{func_name}: 0 <= "weight" <= 1')
        
        match abs(curve):
            case 0:
                expr1 = f'{expr0} x.{means[mean]} {shift} + - x {shift} + + {shift} - 0 {full} clamp'
                expr2 = f'{expr0} y.{means[mean]} {shift} + - y {shift} + + {shift} - 0 {full} clamp'
            case 1:
                expr1 = f'{expr0} x.{means[mean]} {shift} + / x {shift} + * {shift} - 0 {full} clamp'
                expr2 = f'{expr0} y.{means[mean]} {shift} + / y {shift} + * {shift} - 0 {full} clamp'
            case 2:
                expr1 = f'x {shift} + {expr0} log x.{means[mean]} {shift} + log / pow {shift} - 0 {full} clamp'
                expr2 = f'y {shift} + {expr0} log y.{means[mean]} {shift} + log / pow {shift} - 0 {full} clamp'
            case 3:
                expr1 = f'{expr0} 1 x.{means[mean]} {shift} + / pow x {shift} + pow {shift} - 0 {full} clamp'
                expr2 = f'{expr0} 1 y.{means[mean]} {shift} + / pow y {shift} + pow {shift} - 0 {full} clamp'
            case _:
                raise ValueError(f'{func_name}: Please use -3...3 or "None" (only in the list) curve values')
        
        if curve < 0:
            clip = core.std.Invert(clip)
        
        match mode:
            case 0:
                clip = CrazyPlaneStats(core.std.SeparateFields(clip, True), mean, norm=False)
                fields = [clip[::2], clip[1::2]]
                
                match weight:
                    case 0:
                        fields[1] = core.akarin.Expr(fields, expr2)
                    case 1:
                        fields[0] = core.akarin.Expr(fields, expr1)
                    case _:
                        fields[0], fields[1] = core.akarin.Expr(fields, expr1), core.akarin.Expr(fields, expr2)
                
                clip = core.std.Interleave(fields)
                clip = core.std.DoubleWeave(clip, True)[::2]
                clip = core.std.SetFieldBased(clip, 0)
            case 1:
                h = clip.height
                clips = [CrazyPlaneStats(core.std.Crop(clip, 0, 0, i, h - i - 1), mean, norm=False) for i in range(h)]
                
                match weight:
                    case 0:
                        for i in range(1, h, 2):
                            clips[i] = core.akarin.Expr([clips[i - 1], clips[i]], expr2)
                    case 1:
                        for i in range(0, h - 1, 2):
                            clips[i] = core.akarin.Expr([clips[i], clips[i + 1]], expr1)
                    case _:
                        for i in range(0, h - 1, 2):
                            clips[i], clips[i + 1] = (core.akarin.Expr([clips[i], clips[i + 1]], expr1),
                                                      core.akarin.Expr([clips[i], clips[i + 1]], expr2))
                
                clip = core.std.StackVertical(clips)
            case _:
                raise ValueError(f'{func_name}: Please use 0 or 1 mode value')
        
        if curve < 0:
            clip = core.std.Invert(clip)
        
        clip = core.std.RemoveFrameProps(clip, ['minimum', 'maximum', means[mean]])
        
        return clip
    
    match curve:
        case int():
            curve = [curve] * num_p
        case list() if curve:
            if len(curve) < num_p:
                curve += [curve[-1]] * (num_p - len(curve))
            elif len(curve) > num_p:
                raise ValueError(f'{func_name}: "curve" must be shorter or the same length to number of planes, '
                                 'or "curve" must be "int"')
        case _:
            raise ValueError(f'{func_name}: "curve" must be "int" or list[int | None]')
    
    match shift:
        case int():
            shift = [shift * factor] * num_p
        case list() if 0 < len(shift) <= num_p and all(isinstance(i, int) for i in shift):
            shift = [i * factor for i in shift]
            if len(shift) < num_p:
                shift += [shift[-1]] * (num_p - len(shift))
        case _:
            raise ValueError(f'{func_name}: "shift" must be "int" or list[int] and -20 <= "shift" <= 20')
    
    means = ['arithmetic_mean', 'geometric_mean', 'arithmetic_geometric_mean', 'harmonic_mean', 'contraharmonic_mean',
             'root_mean_square', 'root_mean_cube', 'median']
    
    if space == vs.YUV:
        clips = core.std.SplitPlanes(clip)
        
        for i in range(num_p):
            clips[i] = simple_average(clips[i], curve[i], weight, shift[i], mode, mean)
        
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    elif space == vs.GRAY:
        clip = simple_average(clip, curve[0], weight, shift[0], mode, mean)
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    return clip

def nnedi3aas(clip: vs.VideoNode, rg: int = 20, rep: int = 13, clamp: int = 0, planes: int | list[int] | None = None,
              **nnedi3_args: Any) -> vs.VideoNode:
    """nnedi2aas by Didée, ported from AviSynth version with minor additions."""
    func_name = 'nnedi3aas'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    nn = core.znedi3.nnedi3(clip, field=3, planes=planes, **nnedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [0.5 if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes=planes)
    
    if clamp > 0:
        shrpD = core.std.MakeDiff(dbl, Clamp(dbl, RemoveGrain(dbl, [rg if i in planes else 0 for i in range(num_p)]),
                                             dbl, 0, clamp, planes=planes), planes=planes)
    else:
        shrpD = core.std.MakeDiff(dbl, RemoveGrain(dbl, [rg if i in planes else 0 for i in range(num_p)]),
                                  planes=planes)
    
    DD = Repair(shrpD, dblD, [rep if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes=planes)
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else dblD for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def dehalo_mask(clip: vs.VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> vs.VideoNode:
    """
    Fork of jvsfunc.dehalo_mask from dnjulek with minor additions.
    
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.
    
    Args:
        src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
        expand: Expansion of edge mask.
        iterations: Protects parallel lines and corners that are usually damaged by YAHR.
        brz: Adjusts the internal line thickness.
        shift: Corrective shift for fine-tuning iterations
    """
    func_name = 'dehalo_mask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if brz < 0 or brz > 255:
        raise ValueError(f'{func_name}: brz must be between 0 and 255')
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    
    clip = core.std.Expr([clip, core.std.Maximum(clip).std.Maximum()], f'y x - {shift * factor} - 128 *')
    mask = core.tcanny.TCanny(clip, sigma=sqrt(expand * 2), mode=-1).std.Expr('x 16 *')
    
    for _ in range(iterations):
        clip = core.std.Maximum(clip)
    
    for _ in range(iterations):
        clip = core.std.Minimum(clip)
    
    clip = core.std.InvertMask(clip).std.BinarizeMask(80 * factor)
    
    if brz < 255:
        clip = core.std.Inflate(clip).std.Inflate().std.BinarizeMask(brz * factor)
    
    mask = core.std.Expr([mask, RemoveGrain(clip, 12, edges=True)], 'x y min')
    
    if space == vs.YUV:
        mask = core.resize.Point(mask, format=format_id)
    
    return mask

def tp7_deband_mask(clip: vs.VideoNode, thr: float | list[float] = 8, scale: float = 1, rg: bool = True,
                    mt_prewitt: bool = False, **after_args: Any) -> vs.VideoNode:
    
    func_name = 'tp7_deband_mask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    clip = Convolution(clip, 'prewitt', total=1 / scale) if mt_prewitt else core.std.Prewitt(clip, scale=scale)
    
    clip = Binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3), 4)
    
    if space == vs.YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        w = clip.width
        h = clip.height
        
        clips = core.std.SplitPlanes(clip)
        clip = core.std.Expr(clips[1:], 'x y max')
        
        if sub_w > 0 or sub_h > 0:
            bits = clip.format.bits_per_sample
            
            clip = core.fmtc.resample(clip, w, h, kernel='spline', taps=6)
            if bits != 16:
                clip = core.fmtc.bitdepth(clip, bits=bits, dmode=1)
        
        clip = core.std.Expr([clip, clips[0]], 'x y max')
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space == vs.YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def DeHalo_alpha(clip: vs.VideoNode, rx: float = 2.0, ry: float = 2.0, darkstr: float = 1.0, brightstr: float = 1.0,
                 lowsens: float = 50, highsens: float = 50, ss: float = 1.5, showmask: bool = False) -> vs.VideoNode:
    
    func_name = 'DeHalo_alpha'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor
    
    def m4(var: float) -> int:
        return max(int(var / 4 + 0.5) * 4, 16)
    
    halos = core.resize.Bicubic(clip, m4(w / rx), m4(h / ry)).resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    are = Convolution(clip, 'min/max')
    ugly = Convolution(halos, 'min/max')
    so = core.std.Expr(
        [ugly, are],
        f'y x - y 0.001 + / {full - 1} * {lowsens * factor} - y {full} + {full * 2} / {highsens / 100} + *')
    lets = core.std.MaskedMerge(halos, clip, so)
    
    if ss == 1.0:
        remove = Repair(clip, lets, 1)
    else:
        remove = core.resize.Lanczos(clip, x := m4(w * ss), y := m4(h * ss), filter_param_a=3)
        remove = core.std.Expr([remove, core.std.Maximum(lets).resize.Bicubic(x, y)], 'x y min')
        remove = core.std.Expr([remove, core.std.Minimum(lets).resize.Bicubic(x, y)], 'x y max')
        remove = core.resize.Lanczos(remove, w, h, filter_param_a=3)
    
    clip = core.std.Expr([clip, remove], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        clip = core.resize.Point(so, format=orig.format.id) if space == vs.YUV else so
    
    return clip

def FineDehalo(clip: vs.VideoNode, rx: float = 2, ry: float | None = None, thmi: int = 80, thma: int = 128,
               thlimi: int = 50, thlima: int = 100, darkstr: float = 1.0, brightstr: float = 1.0, lowsens: float = 50,
               highsens: float = 50, ss: float = 1.25, showmask: int = 0, contra: float = 0.0, excl: bool = True,
               edgeproc: float = 0.0, mt_prewitt: bool = False) -> vs.VideoNode:
    
    func_name = 'FineDehalo'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    thmi *= factor
    thma *= factor
    thlimi *= factor
    thlima *= factor
    full = 256 * factor - 1
    
    if ry is None:
        ry = rx
    
    rx_i = int(rx + 0.5)
    ry_i = int(ry + 0.5)
    
    dehaloed = DeHalo_alpha(clip, rx, ry, darkstr, brightstr, lowsens, highsens, ss)
    
    if contra > 0:
        half = 128 * factor
        
        bb = RemoveGrain(dehaloed, 11)
        bb2 = Repair(bb, Repair(bb, core.ctmf.CTMF(bb, 2), 1), 1)
        xd = core.std.MakeDiff(bb, bb2).std.Expr(f'x {half} - 2.49 * {contra} * {half} +')
        xdd = core.std.Expr([xd, core.std.MakeDiff(clip, dehaloed)],
                            f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?')
        dehaloed = core.std.MergeDiff(dehaloed, xdd)
    
    edges = Convolution(clip, 'prewitt') if mt_prewitt else core.std.Prewitt(clip)
    
    strong = core.std.Expr(edges, f'x {thmi} - {thma - thmi} / {full} *')
    large = ExpandMulti(strong, sw=rx_i, sh=ry_i)
    light = core.std.Expr(edges, f'x {thlimi} - {thlima - thlimi} / {full} *')
    shrink = ExpandMulti(light, mode='ellipse', sw=rx_i, sh=ry_i).std.Expr('x 4 *')
    shrink = InpandMulti(shrink, mode='ellipse', sw=rx_i, sh=ry_i)
    shrink = RemoveGrain(RemoveGrain(shrink, 20), 20)
    outside = core.std.Expr([large, core.std.Expr([strong, shrink], 'x y max') if excl else strong], 'x y - 2 *')
    
    if edgeproc > 0:
        outside = core.std.Expr([outside, strong], f'x y {edgeproc * 0.66} * +')
    
    outside = core.std.Expr(RemoveGrain(outside, 20), 'x 2 *')
    
    clip = core.std.MaskedMerge(clip, dehaloed, outside)
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        if showmask == 1:
            clip = core.resize.Point(outside, format=orig.format.id) if space == vs.YUV else outside
        elif showmask == 2:
            clip = core.resize.Point(shrink, format=orig.format.id) if space == vs.YUV else shrink
        elif showmask == 3:
            clip = core.resize.Point(edges, format=orig.format.id) if space == vs.YUV else edges
        elif showmask == 4:
            clip = core.resize.Point(strong, format=orig.format.id) if space == vs.YUV else strong
        else:
            raise ValueError(f'{func_name}: Please use 0...4 showmask value')
    
    return clip

def FineDehalo2(clip: vs.VideoNode, hconv: list[int] | None = None, vconv: list[int] | None = None, showmask: bool = False) -> vs.VideoNode:
    
    func_name = 'FineDehalo2'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if hconv is None:
        hconv = [-1, -2, 0, 0, 40, 0, 0, -2, -1]
    
    if vconv is None:
        vconv = [-2, -1, 0, 0, 40, 0, 0, -1, -2]
    
    def grow_mask(clip: vs.VideoNode, mode: str) -> vs.VideoNode:
        
        match mode:
            case 'v':
                coord = [0, 1, 0, 0, 0, 0, 1, 0]
            case 'h':
                coord = [0, 0, 0, 1, 1, 0, 0, 0]
        
        clip = core.std.Maximum(clip, coordinates=coord).std.Minimum(coordinates=coord)
        mask_1 = core.std.Maximum(clip, coordinates=coord)
        mask_2 = core.std.Maximum(mask_1, coordinates=coord).std.Maximum(coordinates=coord)
        clip = core.std.Expr([mask_2, mask_1], 'x y -')
        clip = core.std.Expr(RemoveGrain(clip, 12), 'x 1.8 *')
        
        return clip
    
    fix_h = Convolution(clip, [[1], vconv])
    fix_v = Convolution(clip, [hconv, [1]])
    mask_h = Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], saturate=0, total=4.0)
    mask_v = Convolution(clip, [1, 0, -1, 2, 0, -2, 1, 0, -1], saturate=0, total=4.0)
    temp_h = core.std.Expr([mask_h, mask_v], 'x 3 * y -')
    temp_v = core.std.Expr([mask_v, mask_h], 'x 3 * y -')
    
    mask_h = grow_mask(temp_h, 'v')
    mask_v = grow_mask(temp_v, 'h')
    
    clip = core.std.MaskedMerge(clip, fix_h, mask_h)
    clip = core.std.MaskedMerge(clip, fix_v, mask_v)
    
    if space == vs.YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        clip = core.std.Expr([mask_h, mask_v], 'x y max')
        if space == vs.YUV:
            clip = core.resize.Point(clip, format=orig.format.id)
    
    return clip

def upscaler(clip: vs.VideoNode, dx: int | None = None, dy: int | None = None, mode: int = 0, order: int = 0,
             **upscaler_args: Any) -> vs.VideoNode:
    
    func_name = 'upscaler'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w * 2
    
    if dy is None:
        dy = h * 2
    
    if dx > w * 2 or dy > h * 2:
        raise ValueError(f'{func_name}: upscale size is too big')
    
    def edi3_aa(clip: vs.VideoNode, mode: int, order: bool, **upscaler_args: Any) -> vs.VideoNode:
        
        if order:
            clip = core.std.Transpose(clip)
        
        match mode:
            case 1:
                clip = core.znedi3.nnedi3(clip, field=1, dh=True, **upscaler_args)
                clip = core.std.Transpose(clip)
                clip = core.znedi3.nnedi3(clip, field=1, dh=True, **upscaler_args)
            case 2:
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True, **upscaler_args)
                clip = core.std.Transpose(clip)
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True, **upscaler_args)
            case 3:
                eedi3_keys = signature(core.eedi3m.EEDI3).parameters.keys()
                znedi3_keys = signature(core.znedi3.nnedi3).parameters.keys()
                
                eedi3_args = {key: value for key, value in upscaler_args.items() if key in eedi3_keys}
                znedi3_args = {key: value for key, value in upscaler_args.items() if key in znedi3_keys}
                
                if any((x := i) not in eedi3_args and i not in znedi3_args for i in upscaler_args):
                    raise KeyError(f'{func_name}: Unsupported key {x} in upscaler_args')
                
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True,
                                         sclip=core.znedi3.nnedi3(clip, field=1, dh=True, **znedi3_args), **eedi3_args)
                clip = core.std.Transpose(clip)
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True,
                                         sclip=core.znedi3.nnedi3(clip, field=1, dh=True, **znedi3_args), **eedi3_args)
            case _:
                raise ValueError(f'{func_name}: Please use 0...3 mode value')
        
        if not order:
            clip = core.std.Transpose(clip)
        
        return clip
    
    if mode:
        crop_keys = {'src_left', 'src_top', 'src_width', 'src_height'}
        crop_args = {key: value * 2 for key, value in upscaler_args.items() if key in crop_keys}
        upscaler_args = {key: value for key, value in upscaler_args.items() if key not in crop_keys}
        
        crop_args['src_left'] = crop_args.get('src_left', 0) - 0.5
        crop_args['src_top'] = crop_args.get('src_top', 0) - 0.5
        
        match order:
            case 0:
                clip = edi3_aa(clip, mode, True, **upscaler_args)
            case 1:
                clip = edi3_aa(clip, mode, False, **upscaler_args)
            case 2:
                expr = ['x y max', 'x y min', 'x y max']
                clip = core.std.Expr([edi3_aa(clip, mode, True, **upscaler_args),
                                      edi3_aa(clip, mode, False, **upscaler_args)], expr[:num_p])
            case 3:
                expr = ['x y min', 'x y max', 'x y min']
                clip = core.std.Expr([edi3_aa(clip, mode, True, **upscaler_args),
                                      edi3_aa(clip, mode, False, **upscaler_args)], expr[:num_p])
            case _:
                raise ValueError(f'{func_name}: Please use 0...3 order value')
        
        if clip.format.subsampling_h:
            luma = autotap3(core.std.ShufflePlanes(clip, 0, vs.GRAY), dx, dy, **crop_args)
            crop_args['src_top'] -= 0.5
            clip = core.std.ShufflePlanes([luma, core.resize.Spline36(clip, dx, dy, **crop_args)],
                                          list(range(num_p)), space)
        else:
            clip = autotap3(clip, dx, dy, **crop_args)
    else:
        kernel = upscaler_args.pop('kernel', 'spline36').capitalize()
        clip = getattr(core.resize, kernel)(clip, dx, dy, **upscaler_args)
    
    return clip

def diff_mask(first: vs.VideoNode, second: vs.VideoNode, thr: float = 8, scale: float = 1.0, rg: bool = True,
              mt_prewitt: bool | None = None, **after_args: Any) -> vs.VideoNode:
    
    func_name = 'diff_mask'
    
    if any(not isinstance(i, vs.VideoNode) for i in (first, second)):
        raise TypeError(f'{func_name} both clips must be of the vs.VideoNode type')
    
    if first.format.sample_type != vs.INTEGER or second.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if first.num_frames != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.bits_per_sample != second.format.bits_per_sample:
        raise ValueError(f'{func_name}: Sample types of clips do not match')
    
    space_f = first.format.color_family
    space_s = second.format.color_family
    
    if space_f == vs.YUV:
        format_id = first.format.id
        first = core.std.ShufflePlanes(first, 0, vs.GRAY)
    elif space_f == vs.GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the first clip')
    
    if space_s == vs.YUV:
        second = core.std.ShufflePlanes(second, 0, vs.GRAY)
    elif space_s == vs.GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the second clip')
    
    clip = core.std.Expr([first, second], 'x y - abs')
    
    match mt_prewitt:
        case None:
            clip = core.std.Expr(clip, f'x {scale} *')
        case True:
            clip = Convolution(clip, 'prewitt', total=1 / scale)
        case False:
            clip = core.std.Prewitt(clip, scale=scale)
        case _:
            raise TypeError(f'{func_name}: invalid "mt_prewitt"')
    
    if thr:
        clip = Binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3), 4)
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space_f == vs.YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def apply_range(first: vs.VideoNode, second: vs.VideoNode, /, *args: int | list[int]) -> vs.VideoNode:
    
    func_name = 'apply_range'
    
    if not all(isinstance(i, vs.VideoNode) for i in (first, second)):
        raise TypeError(f'{func_name} both clips must be of the vs.VideoNode type')
    
    num_f = first.num_frames
    
    if num_f != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.name != second.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    for i in args:
        match i:
            case [int(a), int(b)] if 0 <= a < b and b <= num_f:
                if a == 0:
                    first = second[:b] + first[b:]
                elif b == num_f:
                    first = first[:a] + second[a:]
                else:
                    first = first[:a] + second[a:b] + first[b:]
            case int(a) | [int(a)] if 0 <= a < num_f:
                if a == 0:
                    first = second[a] + first[a + 1:]
                elif a == num_f - 1:
                    first = first[:a] + second[a]
                else:
                    first = first[:a] + second[a] + first[a + 1:]
            case _:
                raise ValueError(f'{func_name}: *args must be list[second_frame, first_frame], list[frame] or "int"')
    
    return first

def titles_mask(clip: vs.VideoNode, thr: float = 230, rg: bool = True, **after_args: Any) -> vs.VideoNode:
    
    func_name = 'titles_mask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == vs.YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    elif space == vs.GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    clip = Binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3), 4)
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space == vs.YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def after_mask(clip: vs.VideoNode, boost: bool = False, offset: float = 0.0, flatten: int = 0, borders: list[int] | None = None,
               planes: int | list[int] | None = None, **after_args: int) -> vs.VideoNode:
    
    func_name = 'after_mask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family != vs.GRAY:
        raise TypeError(f'{func_name}: Unsupported color family, only vs.GRAY is supported')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        factor = 1 << clip.format.bits_per_sample - 8
        full = 256 * factor - 1
        expr = f'x {128 * factor} / 0.86 {offset} + pow {full} *'
    else:
        full = 1
        expr = f'x 2 * 0.86 {offset} + pow 1 min 0 max'
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if boost:
        clip = core.std.Expr(clip, expr)
    
    if flatten > 0:
        expr = ['x y max z max' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, flatten + 1):
            clip = core.std.Expr([clip, shift_clip(clip, -i), shift_clip(clip, i)], expr)
    elif flatten < 0:
        expr = ['x y min z min' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, -flatten + 1):
            clip = core.std.Expr([clip, shift_clip(clip, -i), shift_clip(clip, i)], expr)
    
    after_dict = dict(exp_n='Maximum', inp_n='Minimum', def_n='Deflate', inf_n='Inflate')
    
    for key, value in after_args.items():
        if key in after_dict:
            for _ in range(value):
                clip = getattr(core.std, after_dict[key])(clip, planes=planes)
        else:
            raise KeyError(f'{func_name}: Unsupported key {key} in after_args')
    
    if borders:
        if len(borders) < 4:
            defaults = [0, clip.width - 1, 0, clip.height - 1]
            borders += defaults[len(borders):]
        elif len(borders) > 4:
            raise ValueError(f'{func_name}: borders length must be <= 4')
        
        expr = f'X {borders[0]} >= X {borders[1]} <= Y {borders[2]} >= Y {borders[3]} <= and and and {full} 0 ? x min'
        clip = core.std.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def search_field_diffs(clip: vs.VideoNode, mode: int | list[int] = 0, thr: float | list[float] = 0.001, div: float | list[float] = 2.0,
                       norm: bool = True, frames: list[int] | None = None, output: str | None = None, plane: int = 0, mean: int = 0) -> vs.VideoNode:
    """
    Search for deinterlacing failures after ftm/vfm and similar filters, the result is saved to a text file.
    
    The principle of operation is quite simple - each frame is divided into fields and absolute normalized difference
    is calculated for them using two different algorithms.
    
    Args:
        mode: function operation mode.
            0 and 1 - search for frames with absolute normalized difference above the specified threshold.
            2 and 3 - search for the absolute normalized difference change above the specified threshold.
            4 and 5 - search for single anomalies of absolute normalized difference changes above the specified
            threshold (n/p frame is skipped). Of the two possible values, the larger is compared with the threshold.
            The minimum ratio between the anomaly value and the change in adjacent, non-abnormal frames is specified
            by the div parameter.
            6 and 7 - search for double anomalies of absolute normalized difference changes above the specified
            threshold (both n/p frames are skipped). Of the four possible values, the largest is compared with the
            threshold. The minimum ratio between the anomaly value and the change in adjacent, non-abnormal frames
            is specified by the div parameter. In this case, the spread of the values of two abnormal frames must be
            strictly greater than the abnormal value.
            8 and 9 - debug mode for mode 4 and 5.
            10 and 11 - debug mode for mode 6 and 7.
            
            Even modes - the average normalized value is calculated for each field, and then their absolute difference.
            It is well suited for searching combo frames and interlaced fades.
            Odd modes - a classic algorithm, fields are subtracted from each other pixel by modulus,
            and the average normalized value is calculated for the resulting clip.
            It is well suited for detecting temporal anomalies.
            
            You can specify several modes as a list, in which case the result will be sorted by frame number,
            and within one frame by mode. Normal and debug modes cannot be mixed in one list. The default is "0".
        
        thr: the threshold for triggering the mode, it does not work for debug modes.
            You can specify several as a list, they will positionally correspond to the modes.
            If the thr list is less than the list of modes, the last thr value will work for all remaining modes.
            The default is "0.001".
        
        div: sets the minimum ratio between the anomaly value and the change in neighboring, non-abnormal frames.
            It is relevant for modes 4...7. You can specify several as a list, they will positionally correspond to the
            modes. If the div list is less than the list of modes, the last div value will work for all remaining modes.
            The default is "2.0".
        
        norm: normalization to absolute normalized values of the difference between 0 and 1. The default is "True".
        
        frames: a list of frames to check. The default is "all frames".
        
        output: path and name of the output file.
            By default, the file is created in the same directory where the application used for the analysis pass is
            located, the file name is "field_diffs.txt".
        
        plane: the position of the planar for calculating the absolute normalized difference.
            The default is "0" (luminance planar).
    """
    func_name = 'search_field_diffs'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if plane not in set(range(num_p)):
        raise ValueError(f'{func_name}: Unsupported plane')
    
    match mode:
        case int() if mode in set(range(12)):
            mode = [mode]
        case list() if mode and (set(mode) <= set(range(8)) or set(mode) <= set(range(8, 12))):
            pass
        case _:
            raise ValueError(f'{func_name}: Please use 0...7 or 8...11 mode value or list[mode]')
    
    match output:
        case None:
            output = (f'field_diffs_mode({'_'.join(f'{i}' for i in mode)})_'
                      f'thr({'_'.join(f'{i}' for i in thr) if isinstance(thr, list) else thr}).txt')
        case str():
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "output"')
    
    match thr:
        case float():
            thr = [thr] * len(mode)
        case list() if thr and all(isinstance(i, float) for i in thr):
            if len(thr) < len(mode):
                thr += [thr[-1]] * (len(mode) - len(thr))
            elif len(thr) > len(mode):
                raise ValueError(f'{func_name}: len(thr) > len(mode)')
        case _:
            raise TypeError(f'{func_name}: "thr" must be float or list[float]')
    
    match div:
        case float() if div > 1:
            div = [div] * len(mode)
        case list() if div and all(isinstance(i, float) and i > 1 for i in thr):
            if len(div) < len(mode):
                div += [div[-1]] * (len(mode) - len(div))
            elif len(div) > len(mode):
                raise ValueError(f'{func_name}: len(div) > len(mode)')
        case _:
            raise TypeError(f'{func_name}: "div" must be float or list[float] and "div" > 1')
    
    num_f = clip.num_frames
    
    match frames:
        case None:
            frames = list(range(num_f))
        case list() if frames and all(isinstance(i, int) and 0 <= i < num_f for i in frames):
            pass
        case _:
            raise TypeError(f'{func_name}: "frames" must be None or list[0 <= int < num_frames]')
    
    diffs = np.zeros((2, num_f), dtype=np.float64)
    counter = np.full(num_f, np.False_, dtype=np.bool_)
    
    def dump_diffs(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        
        nonlocal diffs, counter
        
        diffs[0][n] = abs(f.props['avg0'] - f.props['avg1'])
        diffs[1][n] = f.props['avg2']
        counter[n] = np.True_
        
        if np.all(counter):
            res = []
            
            dig = max(len(str(num_f)), 5)
            tab = max(len(str(i)) for i in thr)
            
            for i, j in enumerate(mode):
                par = j % 2
                
                match j // 2:
                    case 0:
                        res += [f'{k:>{dig}} {j:>4} {x:.20f} {thr[i]:<{tab}}\n' for k in frames
                                if (x := diffs[par][k]) >= thr[i]]
                    case 1:
                        res += [f'{k:>{dig}} {j:>4} {x:.20f} {thr[i]:<{tab}}\n' for k in frames
                                if (x := abs(diffs[par][max(k - 1, 0)] - diffs[par][k])) >= thr[i]]
                    case 2:
                        res += [f'{k:>{dig}} {j:>4} {x:.20f} {thr[i]:<{tab}} {div[i]}\n' for k in frames
                                if (x := max(abs(diffs[par][max(k - 1, 0)] - diffs[par][k]),
                                abs(diffs[par][k] - diffs[par][min(k + 1, num_f - 1)]))) >= thr[i] and
                                abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 1, num_f - 1)]) <= x / div[i]]
                    case 3:
                        res += [f'{k:>{dig}} {j:>4} {x:.20f} {thr[i]:<{tab}} {div[i]}\n' for k in frames
                                if (x := max(abs(diffs[par][max(k - 1, 0)] - diffs[par][k]),
                                abs(diffs[par][min(k + 1, num_f - 1)] - diffs[par][min(k + 2, num_f - 1)]),
                                abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 1, num_f - 1)]),
                                abs(diffs[par][k] - diffs[par][min(k + 2, num_f - 1)]))) >= thr[i] and
                                abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 2, num_f - 1)]) <= x / div[i]
                                and abs(diffs[par][k] - diffs[par][min(k + 1, num_f - 1)]) > x]
                    case 4:
                        res += [f'{k:>{dig}} {j:>4} {diffs[par][k]:.20f} {(x := max(abs(diffs[par][max(k - 1, 0)] - diffs[par][k]),
                                abs(diffs[par][k] - diffs[par][min(k + 1, num_f - 1)]))):.20f} '
                                f'{min(x / max(abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 1, num_f - 1)]), 1e-20), 99999.99):8.2f}\n'
                                for k in frames]
                    case 5:
                        res += [f'{k:>{dig}} {j:>4} {diffs[par][k]:.20f} {(x := max(abs(diffs[par][max(k - 1, 0)] - diffs[par][k]),
                                abs(diffs[par][min(k + 1, num_f - 1)] - diffs[par][min(k + 2, num_f - 1)]),
                                abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 1, num_f - 1)]),
                                abs(diffs[par][k] - diffs[par][min(k + 2, num_f - 1)]))):.20f} '
                                f'{min(x / max(abs(diffs[par][max(k - 1, 0)] - diffs[par][min(k + 2, num_f - 1)]), 1e-20), 99999.99):8.2f} '
                                f'{abs(diffs[par][k] - diffs[par][min(k + 1, num_f - 1)]):.20f}\n'
                                for k in frames]
            
            if res:
                with open(output, 'w') as file:
                    if set(mode) <= set(range(8, 12)):
                        file.write(f'{'frame':>{dig}} mode {'diff':<22} {'thr':<22} {'div':<8} thr2\n')
                    else:
                        file.write(f'{'frame':>{dig}} mode {'diff':<22} {'thr':<{tab}} div\n')
                    
                    file.writelines(res) if len(mode) == 1 else file.writelines(sorted(res))
            else:
                raise ValueError(f'{func_name}: there is no result, check the settings')
        
        return clip
    
    means = ['arithmetic_mean', 'geometric_mean', 'arithmetic_geometric_mean', 'harmonic_mean', 'contraharmonic_mean',
             'root_mean_square', 'root_mean_cube', 'median']
    
    temp = core.std.SetFieldBased(clip, 0).std.SeparateFields(True)
    fields = [temp[::2], temp[1::2]]
    
    clip = core.akarin.PropExpr([clip, CrazyPlaneStats(fields[0], mean, plane, norm), CrazyPlaneStats(fields[1], mean, plane, norm),
                                 CrazyPlaneStats(core.std.Expr(fields, ['x y - abs' if i == plane else '' for i in range(num_p)]), mean, plane, norm)],
                                lambda: dict(avg0=f'y.{means[mean]}', avg1=f'z.{means[mean]}', avg2=f'a.{means[mean]}'))
    
    clip = core.std.FrameEval(clip, partial(dump_diffs, clip=clip), prop_src=clip, clip_src=clip)
    clip = core.std.RemoveFrameProps(clip, ['avg0', 'avg1', 'avg2'])
    
    return clip

def CombMask2(clip: vs.VideoNode, cthresh: int | None = None, mthresh: int = 9, expand: bool = True, metric: int = 0,
              planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'CombMask2'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if metric not in {0, 1}:
        raise ValueError(f'{func_name}: invalid "metric"')
    
    match cthresh:
        case None:
            cthresh = 10 if metric else 6
        case int() if 0 <= cthresh <= (65535 if metric else 255):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "cthresh"')
    
    if not isinstance(mthresh, int) or mthresh < 0 or mthresh > 255:
        raise ValueError(f'{func_name}: invalid "mthresh"')
    
    if not isinstance(expand, bool):
        raise ValueError(f'{func_name}: invalid "expand"')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor - 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if metric:
        expr = f'x[0,-1] x - x[0,1] x - * {cthresh * factor ** 2} > {full} 0 ?'
    else:
        cthresh *= factor
        expr = (f'x x[0,1] - d1! x x[0,-1] - d2! d1@ {cthresh} > d2@ {cthresh} > and d1@ {-cthresh} < d2@ {-cthresh} < '
                f'and or x[0,2] 4 x * + x[0,-2] + 3 x[0,1] x[0,-1] + * - abs {cthresh * 6} > and {full} 0 ?')
    
    defaults = ['0'] + [f'{128 * factor}'] * (num_p - 1)
    mask = core.akarin.Expr(clip, [expr if i in planes else defaults[i] for i in range(num_p)])
    
    if mthresh:
        expr = f'x y - abs {mthresh * factor} > {full} 0 ?'
        motionmask = core.std.Expr([clip, shift_clip(clip, 1)], [expr if i in planes else defaults[i] for i in range(num_p)])
        
        expr = 'x[0,1] x[0,-1] x max max y min'
        mask = core.akarin.Expr([motionmask, mask], [expr if i in planes else '' for i in range(num_p)])
    
    if expand:
        mask = core.std.Maximum(mask, planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    
    return mask

def MTCombMask(clip: vs.VideoNode, thr1: float = 30, thr2: float = 30, div: float = 256, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'MTCombMask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if thr1 < 0 or thr2 < 0 or thr1 > 65535 or thr2 > 65535:
        raise ValueError(f'{func_name}: Please use 0...65535 thr1 and thr2 value')
    
    if thr1 > thr2:
        raise ValueError(f'{func_name}: thr1 must not be greater than thr2')
    
    if div <= 0:
        raise ValueError(f'{func_name}: div must be greater than zero')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    power = factor ** 2
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = f'x[0,-1] x - x[0,1] x - * var! var@ {thr1 * power} < 0 var@ {thr2 * power} > {256 * factor - 1} var@ {div * factor} / ? ?'
    defaults = ['0'] + [f'{128 * factor}'] * (num_p - 1)
    clip = core.akarin.Expr(clip, [expr if i in planes else defaults[i] for i in range(num_p)])
    
    return clip

def Binarize(clip: vs.VideoNode, thr: float | list[float] = 128, upper: bool = False, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'Binarize'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    match thr:
        case int() | float():
            thr = [thr] * num_p
        case list() if thr:
            if len(thr) < num_p:
                thr += [thr[-1]] * (num_p - len(thr))
            elif len(thr) > num_p:
                raise ValueError(f'{func_name}: "thr" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "thr" must be "float" or "list[float]"')
    
    if upper:
        expr = [f'x {thr[i] * factor} > 0 {256 * factor - 1} ?' for i in range(num_p)]
    else:
        expr = [f'x {thr[i] * factor} > {256 * factor - 1} 0 ?' for i in range(num_p)]
    
    clip = core.std.Expr(clip, [expr[i] if i in planes else f'{128 * factor}' for i in range(num_p)])
    
    return clip

def delcomb(clip: vs.VideoNode, thr1: float = 100, thr2: float = 5, mode: int = 0, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'delcomb'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    mask = MTCombMask(clip, 7, 7, planes=0).std.Deflate(planes=0).std.Deflate(planes=0)
    mask = core.std.Minimum(mask, coordinates=[0, 0, 0, 1, 1, 0, 0, 0], planes=0)
    mask = Binarize(core.std.Maximum(mask, planes=0), thr1, planes=0).std.Maximum(planes=0)
    
    match mode:
        case 0:
            filt = vinverse(clip, 2.3, planes=planes)
        case 1:
            filt = vinverse2(clip, 2.3, planes=planes)
        case 2:
            filt = daa(clip, planes=planes, nns=4, qual=2, pscrn=4, exp=2)
        case _:
            raise ValueError(f'{func_name}: Please use 0...2 "mode" value')
    
    filt = core.std.MaskedMerge(clip, filt, mask, planes=planes, first_plane=True)
    
    clip = core.akarin.Select([clip, filt], CrazyPlaneStats(mask), f'x.arithmetic_mean {thr2 * factor / (256 * factor - 1)} > 1 0 ?')
    
    return clip

def vinverse(clip: vs.VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, clip2: vs.VideoNode | None = None,
                  thr: int = 0, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'vinverse'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if not isinstance(sstr, float):
        raise ValueError(f'{func_name}: invalid "sstr"')
    
    if not isinstance(scl, float):
        raise ValueError(f'{func_name}: invalid "scl"')
    
    if not isinstance(thr, int) or thr < 0 or thr > 255:
        raise ValueError(f'{func_name}: invalid "thr"')
    
    if not isinstance(amnt, int) or amnt < 0 or amnt > 255:
        raise ValueError(f'{func_name}: invalid "amnt"')
    
    Vblur = Convolution(clip, [[1], [50, 99, 50]], planes=planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes=planes)
    
    if clip2 is None:
        Vshrp = Convolution(Vblur, [[1], [1, 4, 6, 4, 1]], planes=planes)
        Vshrp = core.std.Expr([Vblur, Vshrp], [f'x x y - {sstr} * +' if i in planes else '' for i in range(num_p)])
        VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes=planes)
    elif isinstance(clip2, vs.VideoNode) and clip.num_frames == clip2.num_frames and clip.format.name == clip2.format.name:
        VshrpD = core.std.MakeDiff(clip, clip2, planes=planes)
    else:
        raise TypeError(f'{func_name}: invalid "clip2"')
    
    expr = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD = core.std.Expr([VshrpD, VblurD], [expr if i in planes else '' for i in range(num_p)])
    
    res = core.std.MergeDiff(Vblur if clip2 is None else clip2, VlimD, planes=planes)
    
    if thr:
        thr *= factor
        res = core.std.Expr([clip, res, VblurD], [f'z {half} - abs {thr} < x y ?' if i in planes else '' for i in range(num_p)])
    
    match amnt:
        case 255:
            return res
        case 0:
            return clip
        case _:
            amnt *= factor
            expr = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
            return core.std.Expr([clip, res], [expr if i in planes else '' for i in range(num_p)])

def vinverse2(clip: vs.VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'vinverse2'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if not isinstance(sstr, float):
        raise ValueError(f'{func_name}: invalid "sstr"')
    
    if not isinstance(scl, float):
        raise ValueError(f'{func_name}: invalid "scl"')
    
    if not isinstance(amnt, int) or amnt < 0 or amnt > 255:
        raise ValueError(f'{func_name}: invalid "amnt"')
    
    Vblur = sbrV(clip, planes=planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes=planes)
    
    Vshrp = Convolution(Vblur, [[1], [1, 2, 1]], planes=planes)
    Vshrp  = core.std.Expr([Vblur, Vshrp], [f'x x y - {sstr} * +' if i in planes else '' for i in range(num_p)])
    VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes=planes)
    
    expr = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD  = core.std.Expr([VshrpD, VblurD], [expr if i in planes else '' for i in range(num_p)])
    
    res = core.std.MergeDiff(Vblur, VlimD, planes=planes)
    
    match amnt:
        case 255:
            return res
        case 0:
            return clip
        case _:
            amnt *= factor
            expr = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
            return core.std.Expr([clip, res], [expr if i in planes else '' for i in range(num_p)])

def sbr(clip: vs.VideoNode, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'sbr'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        half = 128 << clip.format.bits_per_sample - 8
        expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    else:
        expr = 'x y * 0 < 0 x abs y abs < x y ? ?'
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    rg11 = Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
    rg11D = diff_clamp(core.std.MakeDiff(clip, rg11, planes=planes), planes)
    rg11DD = luma_down(Convolution(luma_up(rg11D, planes), [1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes), planes)
    rg11DD = diff_clamp(core.std.MakeDiff(rg11D, rg11DD, planes=planes), planes)
    rg11DD = core.std.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = clip_clamp(core.std.MakeDiff(clip, rg11DD, planes=planes), planes)
    
    return clip

def sbrV(clip: vs.VideoNode, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'sbrV'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        half = 128 << clip.format.bits_per_sample - 8
        expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    else:
        expr = 'x y * 0 < 0 x abs y abs < x y ? ?'
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    rg11 = Convolution(clip, [[1], [1, 2, 1]], planes=planes)
    rg11D = diff_clamp(core.std.MakeDiff(clip, rg11, planes=planes), planes)
    rg11DD = luma_down(Convolution(luma_up(rg11D, planes), [[1], [1, 2, 1]], planes=planes), planes)
    rg11DD = diff_clamp(core.std.MakeDiff(rg11D, rg11DD, planes=planes), planes)
    rg11DD = core.std.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = clip_clamp(core.std.MakeDiff(clip, rg11DD, planes=planes), planes)
    
    return clip

def Blur(clip: vs.VideoNode, amountH: float = 0, amountV: float | None = None, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'Blur'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if amountV is None:
        amountV = amountH
    
    if amountH < -1 or amountV < -1 or amountH > 1.58 or amountV > 1.58:
        raise ValueError(f'{func_name}: the "amount" allowable range is from -1.0 to +1.58 ')
    
    center_h = 1 / 2 ** amountH
    side_h = (1 - 1 / 2 ** amountH) / 2
    
    center_v = 1 / 2 ** amountV
    side_v = (1 - 1 / 2 ** amountV) / 2
    
    expr = (f'x[-1,-1] x[-1,1] x[1,-1] x[1,1] + + + {side_h * side_v} * x[-1,0] x[1,0] + {side_h * center_v} * + '
            f'x[0,-1] x[0,1] + {center_h * side_v} * + x {center_h * center_v} * + 0 {full} clamp')
    
    clip = chroma_down(core.akarin.Expr(chroma_up(clip, planes), [expr if i in planes else '' for i in range(num_p)]), planes)
    
    return clip

def Sharpen(clip: vs.VideoNode, amountH: float = 0, amountV: float | None = None, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'Sharpen'
    
    if amountV is None:
        amountV = amountH
    
    if amountH < -1.58 or amountV < -1.58 or amountH > 1 or amountV > 1:
        raise ValueError(f'{func_name}: the "amount" allowable range is from -1.58 to +1.0 ')
    
    clip = Blur(clip, -amountH, -amountV, planes)
    
    return clip

def Clamp(clip: vs.VideoNode, bright_limit: vs.VideoNode, dark_limit: vs.VideoNode, overshoot: float = 0, undershoot: float = 0,
          planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'Clamp'
    
    if any(not isinstance(clip, vs.VideoNode) for i in (clip, bright_limit, dark_limit)):
        raise TypeError(f'{func_name} all clips must be of the vs.VideoNode type')
    
    if clip.format.id != bright_limit.format.id or clip.format.id != dark_limit.format.id:
        raise ValueError(f'{func_name}: "clip", "bright_limit" and "dark_limit" must have the same format')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(overshoot, int | float):
        raise TypeError(f'{func_name}: invalid "overshoot"')
    
    if not isinstance(undershoot, int | float):
        raise TypeError(f'{func_name}: invalid "undershoot"')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        factor = 1 << clip.format.bits_per_sample - 8
        expr = f'x y {overshoot * factor} + min z {undershoot * factor} - max'
    else:
        expr = f'x y {overshoot} 255 / + min z {undershoot} 255 / - max 1 min 0 max'
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    clip = chroma_down(core.std.Expr([chroma_up(clip, planes), chroma_up(bright_limit, planes), chroma_up(dark_limit, planes)],
                                     [expr if i in planes else '' for i in range(num_p)]), planes)
    
    return clip

def MinBlur(clip: vs.VideoNode, r: int = 1, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'MinBlur'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        half = 128 << clip.format.bits_per_sample - 8
        expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    else:
        expr = 'x y * 0 < 0 x abs y abs < x y ? ?'
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    match r:
        case 1:
            RG11D = RemoveGrain(clip, [11 if i in planes else 0 for i in range(num_p)])
            RG11D = diff_clamp(core.std.MakeDiff(clip, RG11D, planes=planes), planes)
            
            RG4D = RemoveGrain(clip, [4 if i in planes else 0 for i in range(num_p)])
            RG4D = diff_clamp(core.std.MakeDiff(clip, RG4D, planes=planes), planes)
        case 2:
            RG11D = RemoveGrain(clip, [11 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = diff_clamp(core.std.MakeDiff(clip, RG11D, planes=planes), planes)
            
            RG4D = core.ctmf.CTMF(clip, 2, planes=planes)
            RG4D = diff_clamp(core.std.MakeDiff(clip, RG4D, planes=planes), planes)
        case 3:
            if clip.format.sample_type == vs.FLOAT:
                raise TypeError(f'{func_name}: floating point sample type is not supported with radius=3')
            
            RG11D = RemoveGrain(clip, [11 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = core.std.MakeDiff(clip, RG11D, planes=planes)
            
            RG4D = core.ctmf.CTMF(clip, 3, planes=planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes=planes)
        case _:
            raise ValueError(f'{func_name}: Please use 1...3 "r" value')
    
    DD = core.std.Expr([RG11D, RG4D], [expr if i in planes else '' for i in range(num_p)])
    clip = clip_clamp(core.std.MakeDiff(clip, DD, planes=planes), planes)
    
    return clip

def DitherLumaRebuild(clip: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'DitherLumaRebuild'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    k = (s0 - 1) * c
    t = f'x {16 * factor} - {219 * factor} / 1 min 0 max'
    y = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + {256 * factor} *'
    uv = f'x {half} - 128 * 112 / {half} +'
    
    expr = [y] + [uv] * (num_p - 1)
    clip = core.std.Expr(clip, [expr[i] if i in planes else '' for i in range(num_p)])
    
    return clip

def ExpandMulti(clip: vs.VideoNode, mode: str = 'rectangle', sw: int = 1, sh: int = 1, planes: int | list[int] | None = None,
                    **thr_arg: float) -> vs.VideoNode:
    
    func_name = 'ExpandMulti'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if thr_arg and (len(thr_arg) > 1 or 'threshold' not in thr_arg):
        raise ValueError(f'{func_name}: "thr_arg" must be "threshold=float"')
    
    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and sw % 3 != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None
    
    if mode_m:
        clip = core.std.Maximum(clip, planes=planes, coordinates=mode_m, **thr_arg)
        clip = ExpandMulti(clip, mode=mode, sw=sw - 1, sh=sh - 1, planes=planes, **thr_arg)
    
    return clip

def InpandMulti(clip: vs.VideoNode, mode: str = 'rectangle', sw: int = 1, sh: int = 1, planes: int | list[int] | None = None,
                    **thr_arg: float) -> vs.VideoNode:
    
    func_name = 'InpandMulti'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if thr_arg and (len(thr_arg) > 1 or 'threshold' not in thr_arg):
        raise ValueError(f'{func_name}: "thr_arg" must be "threshold=float"')
    
    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and sw % 3 != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None
    
    if mode_m:
        clip = core.std.Minimum(clip, planes=planes, coordinates=mode_m, **thr_arg)
        clip = InpandMulti(clip, mode=mode, sw=sw - 1, sh=sh - 1, planes=planes, **thr_arg)
    
    return clip

def TemporalSoften(clip: vs.VideoNode, radius: int | None = None, thr: int | None = None, scenechange: int = 0,
                   planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'TemporalSoften'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    match radius:
        case None:
            pass
        case int() if 0 <= radius <= 7:
            pass
        case _:
            raise ValueError(f'{func_name}: Please use 0...7 "radius" value')
    
    if not isinstance(scenechange, int) or scenechange < 0 or scenechange > 255:
        raise ValueError(f'{func_name}: invalid "scenechange"')
    
    match thr:
        case None:
            thr = 256 * factor - 1
        case int() if 0 <= thr <= 255:
            thr *= factor
        case _:
            raise ValueError(f'{func_name}: invalid "thr"')
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    if scenechange:
        clip = SCDetect(clip, scenechange)
    
    def get_smooth(n: int, f: list[vs.VideoFrame], clips: list[vs.VideoNode], core: vs.Core) -> vs.VideoNode:
        
        drop_frames = set()
        
        if scenechange:
            for i in range(radius, 0, -1):
                if f[i].props['_SceneChangeNext'] == 1:
                    drop_frames.update(range(i - 1, -1, -1))
                    break
            
            for i in range(radius, scope - 1):
                if f[i].props['_SceneChangePrev'] == 1:
                    drop_frames.update(range(i + 1, scope))
                    break
        
        expr = f'{' '.join(f'src{radius} src{i} - abs {thr} > src{radius} src{i} ?' if radius != i else
               f'src{i}' for i in range(scope) if i not in drop_frames)} {'+ ' * (scope - len(drop_frames) - 1)}{scope - len(drop_frames)} /'
        clip = core.akarin.Expr(clips, [expr if i in planes else f'src{radius}' for i in range(num_p)])
        
        return clip
    
    if radius:
        scope = radius * 2 + 1
        clips = [shift_clip(clip, i - radius) for i in range(scope)]
        
        clip = core.std.FrameEval(clip, partial(get_smooth, clips=clips, core=core), prop_src=clips, clip_src=clips)
    
    if scenechange:
        clip = core.std.RemoveFrameProps(clip, ['_SceneChangeNext', '_SceneChangePrev'])
    
    return clip

def UnsharpMask(clip: vs.VideoNode, strength: int = 64, radius: int = 3, threshold: int = 8, blur: str = 'box', roundoff: int = 0) -> vs.VideoNode:
    """
    Implementation of UnsharpMask with the ability to select the blur type (box or gauss) and rounding mode.
    
    By default, it perfectly imitates UnsharpMask from the WarpSharp package to AviSynth.
    """
    func_name = 'UnsharpMask'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(strength, int) or strength < 0:
        raise TypeError(f'{func_name}: invalid "strength"')
    
    if not isinstance(radius, int) or radius < 0:
        raise TypeError(f'{func_name}: invalid "radius"')
    
    if not isinstance(threshold, int) or threshold < 0:
        raise TypeError(f'{func_name}: invalid "threshold"')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        factor = 1 << clip.format.bits_per_sample - 8
        threshold *= factor
        full = 256 * factor - 1
    else:
        threshold /= 255
        full = 1
        roundoff = 3
    
    side = radius * 2 + 1
    square = side ** 2
    
    match roundoff:
        case 0:
            rnd = ' trunc'
        case 1:
            rnd = ' 0.5 + trunc'
        case 2:
            rnd = ' round'
        case 3:
            rnd = ''
        case _:
            raise ValueError(f'{func_name}: invalid "roundoff"')
    
    match blur:
        case 'box':
            expr = (f'{' '.join(f'x[{j - radius},{i - radius}]' for i in range(side) for j in range(side))} '
                    f'{'+ ' * (square - 1)}{square} /{rnd} blur! x blur@ - abs {threshold} > x blur@ - {strength} 128 / *{rnd} x + x ? 0 {full} clamp')
        case 'gauss':
            row = [x := (x * (side - i) // i if i != 0 else 1) for i in range(side)]  # noqa: F821, F841
            matrix = [i * j for i in row for j in row]
            expr = (f'{' '.join(f'x[{j - radius},{i - radius}] {matrix[i * side + j]} *' for i in range(side) for j in range(side))} '
                    f'{'+ ' * (square - 1)}{sum(matrix)} /{rnd} blur! x blur@ - abs {threshold} > x blur@ - {strength} 128 / *{rnd} x + x ? 0 {full} clamp')
        case _:
            raise ValueError(f'{func_name}: invalid "blur"')
    
    clip = core.akarin.Expr(clip, [expr] + [''] * (num_p - 1))
    
    return clip

def diff_tfm(clip: vs.VideoNode, nc_clip: vs.VideoNode, ovr_d: str, ovr_c: str, diff_proc: Callable[..., vs.VideoNode] | None = None,
             planes: int | list[int] | None = None, **tfm_args: Any) -> vs.VideoNode:
    
    func_name = 'diff_tfm'
    
    if any(not isinstance(i, vs.VideoNode) for i in (clip, nc_clip)):
        raise TypeError(f'{func_name} both clips must be of the vs.VideoNode type')
    
    if clip.format.name != nc_clip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    num_f = clip.num_frames
    
    if num_f != nc_clip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if any(not isinstance(i, str) for i in (ovr_d, ovr_c)):
        raise TypeError(f'{func_name} both ovr\'s must be of the string type')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    with clip.get_frame(0) as frame:
        if frame.props.get('_FieldBased') in {1, 2}:
            order = frame.props['_FieldBased'] - 1
        else:
            raise KeyError(f'{func_name}: cannot determine field order')
    
    clip_d = core.tivtc.TFM(clip, order=order, ovr=ovr_d, **tfm_args)
    clip_c = core.tivtc.TFM(clip, order=order, ovr=ovr_c, **tfm_args)
    
    nc_clip_d = core.tivtc.TFM(nc_clip, order=order, ovr=ovr_d, **tfm_args)
    nc_clip_c = core.tivtc.TFM(nc_clip, order=order, ovr=ovr_c, **tfm_args)
    
    tfm_args['PP'] = 7
    
    result = ovr_comparator(ovr_d, ovr_c, num_f)
    
    if result[0]:
        clip_d = apply_range(clip_d, core.tivtc.TFM(clip, order=order, ovr=ovr_d, **tfm_args), *result[0])
    
    if result[1]:
        if 'order' in tfm_args:
            tfm_args['order'] ^= 1
        
        if 'field' in tfm_args:
            tfm_args['field'] ^= 1
        
        clip_d = apply_range(clip_d, core.tivtc.TFM(clip, order=order ^ 1, ovr=ovr_d, **tfm_args), *result[1])
    
    diff = [core.std.Expr([clip_c, nc_clip_c], ['x y -' if i in planes else '' for i in range(num_p)]),
            core.std.Expr([clip_c, nc_clip_c], ['y x -' if i in planes else '' for i in range(num_p)])]
    
    match diff_proc:
        case None:
            pass
        case Callable():
            diff = [diff_proc(i, planes=planes) for i in diff]
        case _:
            raise TypeError(f'{func_name} invalid "diff_proc"')
    
    clip = core.std.Expr([nc_clip_d, *diff], ['x y z - +' if i in planes else '' for i in range(num_p)])
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else clip_d for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def diff_transfer(clip: vs.VideoNode, nc_clip: vs.VideoNode, target: vs.VideoNode, diff_proc: Callable[..., vs.VideoNode] | None = None,
                  planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'diff_transfer'
    
    if any(not isinstance(i, vs.VideoNode) for i in (clip, nc_clip, target)):
        raise TypeError(f'{func_name} all clips must be of the vs.VideoNode type')
    
    if clip.format.name != nc_clip.format.name or clip.format.name != target.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != nc_clip.num_frames or clip.num_frames != target.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    diff = [core.std.Expr([clip, nc_clip], ['x y -' if i in planes else '' for i in range(num_p)]),
            core.std.Expr([clip, nc_clip], ['y x -' if i in planes else '' for i in range(num_p)])]
    
    match diff_proc:
        case None:
            pass
        case Callable():
            diff = [diff_proc(i, planes=planes) for i in diff]
        case _:
            raise TypeError(f'{func_name} invalid "diff_proc"')
    
    clip = core.std.Expr([target, *diff], ['x y z - +' if i in planes else '' for i in range(num_p)])
    
    return clip

def shift_clip(clip: vs.VideoNode, shift: int = 0, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'shift_clip'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if not isinstance(shift, int) or abs(shift) >= clip.num_frames:
        raise TypeError(f'{func_name} invalid "shift"')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    orig = clip
    
    if shift > 0:
        clip = clip[0] * shift + clip[:-shift]
    elif shift < 0:
        clip = clip[-shift:] + clip[-1] * -shift
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else orig for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def ovr_comparator(ovr_d: str, ovr_c: str, num_f: int) -> list[list[int]]:
    
    func_name = 'ovr_comparator'
    
    if any(not isinstance(i, str) for i in (ovr_d, ovr_c)):
        raise TypeError(f'{func_name} both ovr\'s must be of the string type')
    
    if not isinstance(num_f, int) or num_f <= 0:
        raise ValueError(f'{func_name}: invalid "num_f"')
    
    frames_d = [None] * num_f
    frames_c = [None] * num_f
    
    with open(ovr_d) as file:
        for line in file:
            if (res := re.search(r'(\d+),(\d+) (\w+)', line)) is not None:
                first = int(res.group(1))
                last = int(res.group(2))
                seq = res.group(3)
                
                for i in range(first, last + 1):
                    frames_d[i] = seq[(i - first) % len(seq)]
                
            elif (res := re.search(r'(\d+) (\w)', line)) is not None:
                frames_d[int(res.group(1))] = res.group(2)
    
    with open(ovr_c) as file:
        for line in file:
            if (res := re.search(r'(\d+),(\d+) (\w+)', line)) is not None:
                first = int(res.group(1))
                last = int(res.group(2))
                seq = res.group(3)
                
                for i in range(first, last + 1):
                    frames_c[i] = seq[(i - first) % len(seq)]
                
            elif (res := re.search(r'(\d+) (\w)', line)) is not None:
                frames_c[int(res.group(1))] = res.group(2)
    
    result = [[], []]
    
    for i in range(num_f):
        match (frames_d[i], frames_c[i]):
            case (None, None) | ('c', 'c') | ('p', 'p') | ('u', 'u'):
                pass
            case ('c', 'p') | ('p', 'c'):
                result[0] += [i]
            case ('c', 'u') | ('p', 'u') | ('u', 'c') | ('u', 'p'):
                result[1] += [i]
            case _:
                raise ValueError(f'{func_name}: undefined behavior in frame {i}')
    
    return result

def RemoveGrain(clip: vs.VideoNode, mode: int | list[int] = 2, edges: bool = False, roundoff: int = 1) -> vs.VideoNode:
    """
    Implementation of RgTools.RemoveGrain with clip edge processing and bank rounding.
    
    Supported modes: -1...28
    
    By default, the reference RemoveGrain is imitated, no edge processing is done (edges=False),
    arithmetic rounding is used (roundoff=1).
    """
    func_name = 'RemoveGrain'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if clip.format.sample_type == vs.INTEGER:
        full = (1 << clip.format.bits_per_sample) - 1
        trnc = ' trunc'
    else:
        full = 1
        trnc = ''
        roundoff = 3
    
    match mode:
        case int() if 0 <= mode <= 28:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and 0 <= i <= 28 for i in mode):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "mode"')
    
    if not isinstance(edges, bool):
        raise TypeError(f'{func_name}: invalid "edges"')
    
    match roundoff:
        case 0:
            rnd = ' trunc'
        case 1:
            rnd = ' 0.5 + trunc'
        case 2:
            rnd = ' round'
        case 3:
            rnd = ''
        case _:
            raise ValueError(f'{func_name}: invalid "roundoff"')
    
    expr = [
        # mode 0
        '',
        # mode 1
        'x x[-1,-1] x[0,-1] min x[1,-1] x[-1,0] min min x[1,0] x[-1,1] min x[0,1] x[1,1] min min min x[-1,-1] x[0,-1] max '
        'x[1,-1] x[-1,0] max max x[1,0] x[-1,1] max x[0,1] x[1,1] max max max clamp',
        # mode 2
        'x x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x[1,0] x[-1,1] x[0,1] x[1,1] sort8 drop swap6 drop5 clamp',
        # mode 3
        'x x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x[1,0] x[-1,1] x[0,1] x[1,1] sort8 drop2 swap5 drop3 swap drop clamp',
        # mode 4
        'x x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x[1,0] x[-1,1] x[0,1] x[1,1] sort8 drop3 swap4 drop swap2 drop2 clamp',
        # mode 5
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! x mil1@ mal1@ clamp '
        'c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp c4! x c1@ - abs d1! x c2@ - abs d2! x c3@ - '
        'abs d3! x c4@ - abs d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 6
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! x mil1@ mal1@ clamp '
        f'c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp c4! x c1@ - abs 2 * mal1@ mil1@ - + {full} '
        f'min d1! x c2@ - abs 2 * mal2@ mil2@ - + {full} min d2! x c3@ - abs 2 * mal3@ mil3@ - + {full} min d3! x c4@ - abs '
        f'2 * mal4@ mil4@ - + {full} min d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = '
        'c3@ c1@ ? ? ?',
        # mode 7
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! x mil1@ mal1@ clamp '
        f'c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp c4! x c1@ - abs mal1@ mil1@ - + {full} min '
        f'd1! x c2@ - abs mal2@ mil2@ - + {full} min d2! x c3@ - abs mal3@ mil3@ - + {full} min d3! x c4@ - abs mal4@ mil4@ '
        f'- + {full} min d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 8
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! x mil1@ mal1@ clamp '
        f'c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp c4! x c1@ - abs mal1@ mil1@ - 2 * + {full} '
        f'min d1! x c2@ - abs mal2@ mil2@ - 2 * + {full} min d2! x c3@ - abs mal3@ mil3@ - 2 * + {full} min d3! x c4@ - abs '
        f'mal4@ mil4@ - 2 * + {full} min d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = '
        'c3@ c1@ ? ? ?',
        # mode 9
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! mal3@ mil3@ - d3! mal4@ mil4@ - d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = x mil4@ mal4@ '
        'clamp mind@ d2@ = x mil2@ mal2@ clamp mind@ d3@ = x mil3@ mal3@ clamp x mil1@ mal1@ clamp ? ? ?',
        # mode 10
        'x x[-1,-1] - abs d1! x x[0,-1] - abs d2! x x[1,-1] - abs d3! x x[-1,0] - abs d4! x x[1,0] - abs d5! x x[-1,1] - abs '
        'd6! x x[0,1] - abs d7! x x[1,1] - abs d8! d1@ d2@ min d3@ min d4@ min d5@ min d6@ min d7@ min d8@ min mind! mind@ '
        'd7@ = x[0,1] mind@ d8@ = x[1,1] mind@ d6@ = x[-1,1] mind@ d2@ = x[0,-1] mind@ d3@ = x[1,-1] mind@ d1@ = x[-1,-1] '
        'mind@ d5@ = x[1,0] x[-1,0] ? ? ? ? ? ? ?',
        # mode 11
        f'x 4 * x[0,-1] x[-1,0] + x[1,0] + x[0,1] + 2 * + x[-1,-1] + x[1,-1] + x[-1,1] + x[1,1] + 16 /{rnd}',
        # mode 12
        f'x 4 * x[0,-1] x[-1,0] + x[1,0] + x[0,1] + 2 * + x[-1,-1] + x[1,-1] + x[-1,1] + x[1,1] + 16 /{rnd}',
        # mode 13
        'x[-1,-1] x[1,1] - abs d1! x[0,-1] x[0,1] - abs d2! x[1,-1] x[-1,1] - abs d3! d1@ d2@ d3@ min min mind! Y 1 bitand 0 '
        f'= mind@ d2@ = x[0,-1] x[0,1] + 2 /{rnd} mind@ d3@ = x[1,-1] x[-1,1] + 2 /{rnd} x[-1,-1] x[1,1] + 2 /{rnd} ? ? x ?',
        # mode 14
        'x[-1,-1] x[1,1] - abs d1! x[0,-1] x[0,1] - abs d2! x[1,-1] x[-1,1] - abs d3! d1@ d2@ d3@ min min mind! Y 1 bitand 1 '
        f'= mind@ d2@ = x[0,-1] x[0,1] + 2 /{rnd} mind@ d3@ = x[1,-1] x[-1,1] + 2 /{rnd} x[-1,-1] x[1,1] + 2 /{rnd} ? ? x ?',
        # mode 15
        f'x[-1,-1] x[0,-1] 2 * + x[1,-1] + x[-1,1] + x[0,1] 2 * + x[1,1] + 8 /{rnd} avg! x[-1,-1] x[1,1] - abs d1! x[0,-1] '
        'x[0,1] - abs d2! x[1,-1] x[-1,1] - abs d3! d1@ d2@ d3@ min min mind! Y 1 bitand 0 = mind@ d2@ = avg@ x[0,-1] x[0,1] '
        'min x[0,-1] x[0,1] max clamp mind@ d3@ = avg@ x[1,-1] x[-1,1] min x[1,-1] x[-1,1] max clamp avg@ x[-1,-1] x[1,1] '
        'min x[-1,-1] x[1,1] max clamp ? ? x ?',
        # mode 16
        f'x[-1,-1] x[0,-1] 2 * + x[1,-1] + x[-1,1] + x[0,1] 2 * + x[1,1] + 8 /{rnd} avg! x[-1,-1] x[1,1] - abs d1! x[0,-1] '
        'x[0,1] - abs d2! x[1,-1] x[-1,1] - abs d3! d1@ d2@ d3@ min min mind! Y 1 bitand 1 = mind@ d2@ = avg@ x[0,-1] x[0,1] '
        'min x[0,-1] x[0,1] max clamp mind@ d3@ = avg@ x[1,-1] x[-1,1] min x[1,-1] x[-1,1] max clamp avg@ x[-1,-1] x[1,1] '
        'min x[-1,-1] x[1,1] max clamp ? ? x ?',
        # mode 17
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! mil1@ mil2@ max mil3@ '
        'max mil4@ max lower! mal1@ mal2@ min mal3@ min mal4@ min upper! x lower@ upper@ min lower@ upper@ max clamp',
        # mode 18
        'x x[-1,-1] - abs x x[1,1] - abs max d1! x x[0,-1] - abs x x[0,1] - abs max d2! x x[1,-1] - abs x x[-1,1] - abs max '
        'd3! x x[-1,0] - abs x x[1,0] - abs max d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = x x[-1,0] x[1,0] min '
        'x[-1,0] x[1,0] max clamp mind@ d2@ = x x[0,-1] x[0,1] min x[0,-1] x[0,1] max clamp mind@ d3@ = x x[1,-1] x[-1,1] '
        'min x[1,-1] x[-1,1] max clamp x x[-1,-1] x[1,1] min x[-1,-1] x[1,1] max clamp ? ? ?',
        # mode 19
        f'x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 8 /{rnd}',
        # mode 20
        f'x[-1,-1] x[0,-1] + x[1,-1] + x[-1,0] + x + x[1,0] + x[-1,1] + x[0,1] + x[1,1] + 9 /{rnd}',
        # mode 21
        f'x x[-1,-1] x[1,1] + 2 /{trnc} x[0,-1] x[0,1] + 2 /{trnc} min x[1,-1] x[-1,1] + 2 /{trnc} min x[-1,0] x[1,0] + 2 /'
        f'{trnc} min x[-1,-1] x[1,1] + 2 /{rnd} x[0,-1] x[0,1] + 2 /{rnd} max x[1,-1] x[-1,1] + 2 /{rnd} max x[-1,0] x[1,0] + '
        f'2 /{rnd} max clamp',
        # mode 22
        f'x[-1,-1] x[1,1] + 2 /{rnd} l1! x[0,-1] x[0,1] + 2 /{rnd} l2! x[1,-1] x[-1,1] + 2 /{rnd} l3! x[-1,0] x[1,0] + 2 '
        f'/{rnd} l4! x l1@ l2@ min l3@ min l4@ min l1@ l2@ max l3@ max l4@ max clamp',
        # mode 23
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! mal1@ mil1@ - ld1! '
        'mal2@ mil2@ - ld2! mal3@ mil3@ - ld3! mal4@ mil4@ - ld4! x x mal1@ - ld1@ min x mal2@ - ld2@ min max x mal3@ - ld3@ '
        'min max x mal4@ - ld4@ min max 0 max - 0 max mil1@ x - ld1@ min mil2@ x - ld2@ min max mil3@ x - ld3@ min max mil4@ '
        f'x - ld4@ min max 0 max + {full} min',
        # mode 24
        'x[-1,-1] x[1,1] max mal1! x[-1,-1] x[1,1] min mil1! x[0,-1] x[0,1] max mal2! x[0,-1] x[0,1] min mil2! x[1,-1] '
        'x[-1,1] max mal3! x[1,-1] x[-1,1] min mil3! x[-1,0] x[1,0] max mal4! x[-1,0] x[1,0] min mil4! mal1@ mil1@ - ld1! '
        'mal2@ mil2@ - ld2! mal3@ mil3@ - ld3! mal4@ mil4@ - ld4! x mal1@ - 0 max u1! x mal2@ - 0 max u2! x mal3@ - 0 max '
        'u3! x mal4@ - 0 max u4! mil1@ x - 0 max d1! mil2@ x - 0 max d2! mil3@ x - 0 max d3! mil4@ x - 0 max d4! x u1@ ld1@ '
        'u1@ - min u2@ ld2@ u2@ - min max u3@ ld3@ u3@ - min max u4@ ld4@ u4@ - min max 0 max - 0 max d1@ ld1@ d1@ - min d2@ '
        f'ld2@ d2@ - min max d3@ ld3@ d3@ - min max d4@ ld4@ d4@ - min max 0 max + {full} min',
        # mode 25
        f'x x[-1,0] < {full} x x[-1,0] - ? x x[1,0] < {full} x x[1,0] - ? min x x[-1,-1] < {full} x x[-1,-1] - ? min x '
        f'x[0,-1] < {full} x x[0,-1] - ? min x x[1,-1] < {full} x x[1,-1] - ? min x x[-1,1] < {full} x x[-1,1] - ? min x '
        f'x[0,1] < {full} x x[0,1] - ? min x x[1,1] < {full} x x[1,1] - ? min mn! x[-1,0] x < {full} x[-1,0] x - ? x[1,0] x '
        f'< {full} x[1,0] x - ? min x[-1,-1] x < {full} x[-1,-1] x - ? min x[0,-1] x < {full} x[0,-1] x - ? min x[1,-1] x < '
        f'{full} x[1,-1] x - ? min x[-1,1] x < {full} x[-1,1] x - ? min x[0,1] x < {full} x[0,1] x - ? min x[1,1] x < {full} '
        f'x[1,1] x - ? min pl! x pl@ 2 /{trnc} mn@ pl@ - 0 max min + {full} min mn@ 2 /{trnc} pl@ mn@ - 0 max min - 0 max',
        # mode 26
        'x[-1,-1] x[0,-1] min x[0,-1] x[1,-1] min max x[1,-1] x[1,0] min max x[1,0] x[1,1] min max x[0,1] x[1,1] min x[-1,1] '
        'x[0,1] min max x[-1,0] x[-1,1] min max x[-1,-1] x[-1,0] min max max lower! x[-1,-1] x[0,-1] max x[0,-1] x[1,-1] max '
        'min x[1,-1] x[1,0] max min x[1,0] x[1,1] max min x[0,1] x[1,1] max x[-1,1] x[0,1] max min x[-1,0] x[-1,1] max min '
        'x[-1,-1] x[-1,0] max min min upper! x lower@ upper@ min lower@ upper@ max clamp',
        # mode 27
        'x[-1,-1] x[1,1] min x[-1,-1] x[0,-1] min max x[0,1] x[1,1] min max x[0,-1] x[0,1] min max x[0,-1] x[1,-1] min '
        'x[-1,1] x[0,1] min max x[1,-1] x[-1,1] min max x[1,-1] x[1,0] min max max x[-1,0] x[-1,1] min x[-1,0] x[1,0] min '
        'max x[1,0] x[1,1] min max x[-1,-1] x[-1,0] min max max lower! x[-1,-1] x[1,1] max x[-1,-1] x[0,-1] max min x[0,1] '
        'x[1,1] max min x[0,-1] x[0,1] max min x[0,-1] x[1,-1] max x[-1,1] x[0,1] max min x[1,-1] x[-1,1] max min x[1,-1] '
        'x[1,0] max min min x[-1,0] x[-1,1] max x[-1,0] x[1,0] max min x[1,0] x[1,1] max min x[-1,-1] x[-1,0] max min min '
        'upper! x lower@ upper@ min lower@ upper@ max clamp',
        # mode 28
        'x[-1,-1] x[0,-1] min x[0,-1] x[1,-1] min max x[1,-1] x[1,0] min max x[1,0] x[1,1] min max x[0,1] x[1,1] min x[-1,1] '
        'x[0,1] min max x[-1,0] x[-1,1] min max x[-1,-1] x[-1,0] min max max x[-1,-1] x[1,1] min x[1,-1] x[-1,1] min max '
        'x[0,-1] x[0,1] min max x[-1,0] x[1,0] min max max lower! x[-1,-1] x[0,-1] max x[0,-1] x[1,-1] max min x[1,-1] '
        'x[1,0] max min x[1,0] x[1,1] max min x[0,1] x[1,1] max x[-1,1] x[0,1] max min x[-1,0] x[-1,1] max min x[-1,-1] '
        'x[-1,0] max min min x[-1,-1] x[1,1] max x[1,-1] x[-1,1] max min x[0,-1] x[0,1] max min x[-1,0] x[1,0] max min min '
        'upper! x lower@ upper@ min lower@ upper@ max clamp'
    ]
    
    orig = clip
    
    planes = [i for i, j in enumerate(mode + [mode[-1]] * (num_p - len(mode))) if j]
    clip = chroma_down(core.akarin.Expr(chroma_up(clip, planes), [expr[i] for i in mode]), planes)
    
    if not edges:
        expr = 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?'
        clip = core.akarin.Expr([clip, orig], [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def Repair(clip: vs.VideoNode, refclip: vs.VideoNode, mode: int | list[int] = 2, edges: bool = False) -> vs.VideoNode:
    """
    Implementation of RgTools.Repair with clip edge processing.
    
    Supported modes: -1...28
    
    By default, the reference Repair is imitated, no edge processing is done (edges=False).
    """
    func_name = 'Repair'
    
    if any(not isinstance(i, vs.VideoNode) for i in (clip, refclip)):
        raise TypeError(f'{func_name} both clips must be of the vs.VideoNode type')
    
    if clip.format.name != refclip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != refclip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match mode:
        case int() if 0 <= mode <= 28:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and 0 <= i <= 28 for i in mode):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "mode"')
    
    if not isinstance(edges, bool):
        raise TypeError(f'{func_name}: invalid "edges"')
    
    expr = [
        # mode 0
        '',
        # mode 1
        'x y[-1,-1] y[0,-1] min y[1,-1] y[-1,0] min min y[1,0] y[-1,1] min y[0,1] y[1,1] min min min y min y[-1,-1] y[0,-1] '
        'max y[1,-1] y[-1,0] max max y[1,0] y[-1,1] max y[0,1] y[1,1] max max max y max clamp',
        # mode 2
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y y[1,0] y[-1,1] y[0,1] y[1,1] sort9 drop swap7 drop6 clamp',
        # mode 3
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y y[1,0] y[-1,1] y[0,1] y[1,1] sort9 drop2 swap6 drop4 swap drop clamp',
        # mode 4
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y y[1,0] y[-1,1] y[0,1] y[1,1] sort9 drop3 swap5 drop2 swap2 drop2 clamp',
        # mode 5
        'y[-1,-1] y[1,1] max y max mal1! y[-1,-1] y[1,1] min y min mil1! y[0,-1] y[0,1] max y max mal2! y[0,-1] y[0,1] min y '
        'min mil2! y[1,-1] y[-1,1] max y max mal3! y[1,-1] y[-1,1] min y min mil3! y[-1,0] y[1,0] max y max mal4! y[-1,0] '
        'y[1,0] min y min mil4! x mil1@ mal1@ clamp c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp '
        'c4! x c1@ - abs d1! x c2@ - abs d2! x c3@ - abs d3! x c4@ - abs d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = '
        'c4@ mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 6
        'y[-1,-1] y[1,1] max y max mal1! y[-1,-1] y[1,1] min y min mil1! y[0,-1] y[0,1] max y max mal2! y[0,-1] y[0,1] min y '
        'min mil2! y[1,-1] y[-1,1] max y max mal3! y[1,-1] y[-1,1] min y min mil3! y[-1,0] y[1,0] max y max mal4! y[-1,0] '
        'y[1,0] min y min mil4! x mil1@ mal1@ clamp c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp '
        f'c4! x c1@ - abs 2 * mal1@ mil1@ - + {full} min d1! x c2@ - abs 2 * mal2@ mil2@ - + {full} min d2! x c3@ - abs 2 * '
        f'mal3@ mil3@ - + {full} min d3! x c4@ - abs 2 * mal4@ mil4@ - + {full} min d4! d1@ d2@ min d3@ min d4@ min mind! '
        'mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 7
        'y[-1,-1] y[1,1] max y max mal1! y[-1,-1] y[1,1] min y min mil1! y[0,-1] y[0,1] max y max mal2! y[0,-1] y[0,1] min y '
        'min mil2! y[1,-1] y[-1,1] max y max mal3! y[1,-1] y[-1,1] min y min mil3! y[-1,0] y[1,0] max y max mal4! y[-1,0] '
        'y[1,0] min y min mil4! x mil1@ mal1@ clamp c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp '
        f'c4! x c1@ - abs mal1@ mil1@ - + {full} min d1! x c2@ - abs mal2@ mil2@ - + {full} min d2! x c3@ - abs mal3@ mil3@ '
        f'- + {full} min d3! x c4@ - abs mal4@ mil4@ - + {full} min d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = c4@ '
        'mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 8
        'y[-1,-1] y[1,1] max y max mal1! y[-1,-1] y[1,1] min y min mil1! y[0,-1] y[0,1] max y max mal2! y[0,-1] y[0,1] min y '
        'min mil2! y[1,-1] y[-1,1] max y max mal3! y[1,-1] y[-1,1] min y min mil3! y[-1,0] y[1,0] max y max mal4! y[-1,0] '
        'y[1,0] min y min mil4! x mil1@ mal1@ clamp c1! x mil2@ mal2@ clamp c2! x mil3@ mal3@ clamp c3! x mil4@ mal4@ clamp '
        f'c4! x c1@ - abs mal1@ mil1@ - 2 * + {full} min d1! x c2@ - abs mal2@ mil2@ - 2 * + {full} min d2! x c3@ - abs '
        f'mal3@ mil3@ - 2 * + {full} min d3! x c4@ - abs mal4@ mil4@ - 2 * + {full} min d4! d1@ d2@ min d3@ min d4@ min '
        'mind! mind@ d4@ = c4@ mind@ d2@ = c2@ mind@ d3@ = c3@ c1@ ? ? ?',
        # mode 9
        'y[-1,-1] y[1,1] max y max mal1! y[-1,-1] y[1,1] min y min mil1! y[0,-1] y[0,1] max y max mal2! y[0,-1] y[0,1] min y '
        'min mil2! y[1,-1] y[-1,1] max y max mal3! y[1,-1] y[-1,1] min y min mil3! y[-1,0] y[1,0] max y max mal4! y[-1,0] '
        'y[1,0] min y min mil4! mal1@ mil1@ - d1! mal2@ mil2@ - d2! mal3@ mil3@ - d3! mal4@ mil4@ - d4! d1@ d2@ min d3@ min '
        'd4@ min mind! mind@ d4@ = x mil4@ mal4@ clamp mind@ d2@ = x mil2@ mal2@ clamp mind@ d3@ = x mil3@ mal3@ clamp x '
        'mil1@ mal1@ clamp ? ? ?',
        # mode 10
        'x y[-1,-1] - abs d1! x y[0,-1] - abs d2! x y[1,-1] - abs d3! x y[-1,0] - abs d4! x y[1,0] - abs d5! x y[-1,1] - abs '
        'd6! x y[0,1] - abs d7! x y[1,1] - abs d8! x y - abs dy! d1@ d2@ min d3@ min d4@ min d5@ min d6@ min d7@ min d8@ min '
        'dy@ min mind! mind@ d7@ = y[0,1] mind@ d8@ = y[1,1] mind@ d6@ = y[-1,1] mind@ d2@ = y[0,-1] mind@ d3@ = y[1,-1] '
        'mind@ d1@ = y[-1,-1] mind@ d5@ = y[1,0] mind@ dy@ = y y[-1,0] ? ? ? ? ? ? ? ?',
        # mode 11
        'x y[-1,-1] y[0,-1] min y[1,-1] y[-1,0] min min y[1,0] y[-1,1] min y[0,1] y[1,1] min min min y min y[-1,-1] y[0,-1] '
        'max y[1,-1] y[-1,0] max max y[1,0] y[-1,1] max y[0,1] y[1,1] max max max y max clamp',
        # mode 12
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y[1,0] y[-1,1] y[0,1] y[1,1] sort8 drop y min swap6 drop5 y max clamp',
        # mode 13
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y[1,0] y[-1,1] y[0,1] y[1,1] sort8 drop2 y min swap5 drop3 y max swap drop clamp',
        # mode 14
        'x y[-1,-1] y[0,-1] y[1,-1] y[-1,0] y[1,0] y[-1,1] y[0,1] y[1,1] sort8 drop3 y min swap4 drop y max swap2 drop2 clamp',
        # mode 15
        'y[-1,-1] y[1,1] max mal1! y[-1,-1] y[1,1] min mil1! y[0,-1] y[0,1] max mal2! y[0,-1] y[0,1] min mil2! y[1,-1] '
        'y[-1,1] max mal3! y[1,-1] y[-1,1] min mil3! y[-1,0] y[1,0] max mal4! y[-1,0] y[1,0] min mil4! y y mil1@ mal1@ clamp '
        '- abs d1! y y mil2@ mal2@ clamp - abs d2! y y mil3@ mal3@ clamp - abs d3! y y mil4@ mal4@ clamp - abs d4! d1@ d2@ '
        'min d3@ min d4@ min mind! mind@ d4@ = x mil4@ y min mal4@ y max clamp mind@ d2@ = x mil2@ y min mal2@ y max clamp '
        'mind@ d3@ = x mil3@ y min mal3@ y max clamp x mil1@ y min mal1@ y max clamp ? ? ?',
        # mode 16
        'y[-1,-1] y[1,1] max mal1! y[-1,-1] y[1,1] min mil1! y[0,-1] y[0,1] max mal2! y[0,-1] y[0,1] min mil2! y[1,-1] '
        'y[-1,1] max mal3! y[1,-1] y[-1,1] min mil3! y[-1,0] y[1,0] max mal4! y[-1,0] y[1,0] min mil4! y y mil1@ mal1@ clamp '
        f'- abs 2 * mal1@ mil1@ - + {full} min d1! y y mil2@ mal2@ clamp - abs 2 * mal2@ mil2@ - + {full} min d2! y y mil3@ '
        f'mal3@ clamp - abs 2 * mal3@ mil3@ - + {full} min d3! y y mil4@ mal4@ clamp - abs 2 * mal4@ mil4@ - + {full} min '
        'd4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = x mil4@ y min mal4@ y max clamp mind@ d2@ = x mil2@ y min mal2@ y '
        'max clamp mind@ d3@ = x mil3@ y min mal3@ y max clamp x mil1@ y min mal1@ y max clamp ? ? ?',
        # mode 17
        'y[-1,-1] y[1,1] min y[0,-1] y[0,1] min max y[1,-1] y[-1,1] min y[-1,0] y[1,0] min max max lower! y[-1,-1] y[1,1] '
        'max y[0,-1] y[0,1] max min y[1,-1] y[-1,1] max y[-1,0] y[1,0] max min min upper! x lower@ upper@ min y min lower@ '
        'upper@ max y max clamp',
        # mode 18
        'y y[-1,-1] - abs y y[1,1] - abs max d1! y y[0,-1] - abs y y[0,1] - abs max d2! y y[1,-1] - abs y y[-1,1] - abs max '
        'd3! y y[-1,0] - abs y y[1,0] - abs max d4! d1@ d2@ min d3@ min d4@ min mind! mind@ d4@ = x y[-1,0] y[1,0] min y min '
        'y[-1,0] y[1,0] max y max clamp mind@ d2@ = x y[0,-1] y[0,1] min y min y[0,-1] y[0,1] max y max clamp mind@ d3@ = x '
        'y[1,-1] y[-1,1] min y min y[1,-1] y[-1,1] max y max clamp x y[-1,-1] y[1,1] min y min y[-1,-1] y[1,1] max y max '
        'clamp ? ? ?',
        # mode 19
        'y y[-1,-1] - abs y y[0,-1] - abs min y y[1,-1] - abs min y y[-1,0] - abs min y y[1,0] - abs min y y[-1,1] - abs min '
        f'y y[0,1] - abs min y y[1,1] - abs min mind! x y mind@ - 0 max y mind@ + {full} min clamp',
        # mode 20
        'y y[-1,-1] - abs d1! y y[0,-1] - abs d2! y y[1,-1] - abs d3! y y[-1,0] - abs d4! y y[1,0] - abs d5! y y[-1,1] - abs '
        'd6! y y[0,1] - abs d7! y y[1,1] - abs d8! d1@ d2@ min mind2! mind2@ d3@ min mind3! mind3@ d4@ min mind4! mind4@ d5@ '
        'min mind5! mind5@ d6@ min mind6! mind6@ d7@ min mind7! d1@ d2@ max mind2@ d3@ clamp mind3@ d4@ clamp mind4@ d5@ '
        f'clamp mind5@ d6@ clamp mind6@ d7@ clamp mind7@ d8@ clamp maxd! x y maxd@ - 0 max y maxd@ + {full} min clamp',
        # mode 21
        'y[-1,-1] y[1,1] max y - 0 max y y[-1,-1] y[1,1] min - 0 max max y[0,-1] y[0,1] max y - 0 max y y[0,-1] y[0,1] min - '
        '0 max max min y[1,-1] y[-1,1] max y - 0 max y y[1,-1] y[-1,1] min - 0 max max min y[-1,0] y[1,0] max y - 0 max y '
        f'y[-1,0] y[1,0] min - 0 max max min minu! x y minu@ - 0 max y minu@ + {full} min clamp',
        # mode 22
        'x y[-1,-1] - abs x y[0,-1] - abs min x y[1,-1] - abs min x y[-1,0] - abs min x y[1,0] - abs min x y[-1,1] - abs min '
        f'x y[0,1] - abs min x y[1,1] - abs min mind! y x mind@ - 0 max x mind@ + {full} min clamp',
        # mode 23
        'x y[-1,-1] - abs d1! x y[0,-1] - abs d2! x y[1,-1] - abs d3! x y[-1,0] - abs d4! x y[1,0] - abs d5! x y[-1,1] - abs '
        'd6! x y[0,1] - abs d7! x y[1,1] - abs d8! d1@ d2@ min mind2! mind2@ d3@ min mind3! mind3@ d4@ min mind4! mind4@ d5@ '
        'min mind5! mind5@ d6@ min mind6! mind6@ d7@ min mind7! d1@ d2@ max mind2@ d3@ clamp mind3@ d4@ clamp mind4@ d5@ '
        f'clamp mind5@ d6@ clamp mind6@ d7@ clamp mind7@ d8@ clamp maxd! y x maxd@ - 0 max x maxd@ + {full} min clamp',
        # mode 24
        'y[-1,-1] y[1,1] max x - 0 max x y[-1,-1] y[1,1] min - 0 max max y[0,-1] y[0,1] max x - 0 max x y[0,-1] y[0,1] min - '
        '0 max max min y[1,-1] y[-1,1] max x - 0 max x y[1,-1] y[-1,1] min - 0 max max min y[-1,0] y[1,0] max x - 0 max x '
        f'y[-1,0] y[1,0] min - 0 max max min minu! y x minu@ - 0 max x minu@ + {full} min clamp',
        # mode 25
        '',
        # mode 26
        'y[-1,-1] y[0,-1] min y[0,-1] y[1,-1] min max y[1,-1] y[1,0] min max y[1,0] y[1,1] min max y[0,1] y[1,1] min y[-1,1] '
        'y[0,1] min max y[-1,0] y[-1,1] min max y[-1,-1] y[-1,0] min max max lower! y[-1,-1] y[0,-1] max y[0,-1] y[1,-1] max '
        'min y[1,-1] y[1,0] max min y[1,0] y[1,1] max min y[0,1] y[1,1] max y[-1,1] y[0,1] max min y[-1,0] y[-1,1] max min '
        'y[-1,-1] y[-1,0] max min min upper! x lower@ upper@ min y min lower@ upper@ max y max clamp',
        # mode 27
        'y[-1,-1] y[1,1] min y[-1,-1] y[0,-1] min max y[0,1] y[1,1] min max y[0,-1] y[0,1] min max y[0,-1] y[1,-1] min '
        'y[-1,1] y[0,1] min max y[1,-1] y[-1,1] min max y[1,-1] y[1,0] min max max y[-1,0] y[-1,1] min y[-1,0] y[1,0] min '
        'max y[1,0] y[1,1] min max y[-1,-1] y[-1,0] min max max lower! y[-1,-1] y[1,1] max y[-1,-1] y[0,-1] max min y[0,1] '
        'y[1,1] max min y[0,-1] y[0,1] max min y[0,-1] y[1,-1] max y[-1,1] y[0,1] max min y[1,-1] y[-1,1] max min y[1,-1] '
        'y[1,0] max min min y[-1,0] y[-1,1] max y[-1,0] y[1,0] max min y[1,0] y[1,1] max min y[-1,-1] y[-1,0] max min min '
        'upper! x lower@ upper@ min y min lower@ upper@ max y max clamp',
        # mode 28
        'y[-1,-1] y[0,-1] min y[0,-1] y[1,-1] min max y[1,-1] y[1,0] min max y[1,0] y[1,1] min max y[0,1] y[1,1] min y[-1,1] '
        'y[0,1] min max y[-1,0] y[-1,1] min max y[-1,-1] y[-1,0] min max max y[-1,-1] y[1,1] min y[1,-1] y[-1,1] min max '
        'y[0,-1] y[0,1] min max y[-1,0] y[1,0] min max max lower! y[-1,-1] y[0,-1] max y[0,-1] y[1,-1] max min y[1,-1] '
        'y[1,0] max min y[1,0] y[1,1] max min y[0,1] y[1,1] max y[-1,1] y[0,1] max min y[-1,0] y[-1,1] max min y[-1,-1] '
        'y[-1,0] max min min y[-1,-1] y[1,1] max y[1,-1] y[-1,1] max min y[0,-1] y[0,1] max min y[-1,0] y[1,0] max min min '
        'upper! x lower@ upper@ min y min lower@ upper@ max y max clamp'
    ]
    
    orig = clip
    
    planes = [i for i, j in enumerate(mode + [mode[-1]] * (num_p - len(mode))) if j]
    clip = chroma_down(core.akarin.Expr([chroma_up(clip, planes), chroma_up(refclip, planes)], [expr[i] for i in mode]), planes)
    
    if not edges:
        expr = 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?'
        clip = core.akarin.Expr([clip, orig], [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def TemporalRepair(clip: vs.VideoNode, refclip: vs.VideoNode, mode: int = 0, edges: bool = False, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'TemporalRepair'
    
    if any(not isinstance(i, vs.VideoNode) for i in (clip, refclip)):
        raise TypeError(f'{func_name} both clips must be of the vs.VideoNode type')
    
    if clip.format.name != refclip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != refclip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(mode, int) or mode < 0 or mode > 4:
        raise ValueError(f'{func_name}: invalid "mode"')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = [
        # mode 0
        'x a z min y min a z max y max clamp',
        # mode 1
        'x y y[-1,-1] a[-1,-1] z[-1,-1] min - 0 max y[-1,0] a[-1,0] z[-1,0] min - 0 max max y[-1,1] a[-1,1] z[-1,1] min - 0 '
        'max max y[1,-1] a[1,-1] z[1,-1] min - 0 max max y[1,0] a[1,0] z[1,0] min - 0 max max y[1,1] a[1,1] z[1,1] min - 0 '
        'max max y[0,-1] a[0,-1] z[0,-1] min - 0 max max y[0,1] a[0,1] z[0,1] min - 0 max max - 0 max z min a min a[-1,-1] '
        'z[-1,-1] max y[-1,-1] - 0 max a[-1,0] z[-1,0] max y[-1,0] - 0 max max a[-1,1] z[-1,1] max y[-1,1] - 0 max max '
        'a[1,-1] z[1,-1] max y[1,-1] - 0 max max a[1,0] z[1,0] max y[1,0] - 0 max max a[1,1] z[1,1] max y[1,1] - 0 max max '
        f'a[0,-1] z[0,-1] max y[0,-1] - 0 max max a[0,1] z[0,1] max y[0,1] - 0 max max y + {full} min z max a max clamp',
        # mode 2
        'y[-1,-1] a[-1,-1] z[-1,-1] min - 0 max y[-1,0] a[-1,0] z[-1,0] min - 0 max max y[-1,1] a[-1,1] z[-1,1] min - 0 max '
        'max y[1,-1] a[1,-1] z[1,-1] min - 0 max max y[1,0] a[1,0] z[1,0] min - 0 max max y[1,1] a[1,1] z[1,1] min - 0 max '
        'max y[0,-1] a[0,-1] z[0,-1] min - 0 max max y[0,1] a[0,1] z[0,1] min - 0 max max y a z min - 0 max max a[-1,-1] '
        'z[-1,-1] max y[-1,-1] - 0 max a[-1,0] z[-1,0] max y[-1,0] - 0 max max a[-1,1] z[-1,1] max y[-1,1] - 0 max max '
        'a[1,-1] z[1,-1] max y[1,-1] - 0 max max a[1,0] z[1,0] max y[1,0] - 0 max max a[1,1] z[1,1] max y[1,1] - 0 max max '
        'a[0,-1] z[0,-1] max y[0,-1] - 0 max max a[0,1] z[0,1] max y[0,1] - 0 max max a z max y - 0 max max max ulmax! x y '
        f'ulmax@ - 0 max y ulmax@ + {full} min clamp',
        # mode 3
        'y[-1,-1] a[-1,-1] - abs y[-1,0] a[-1,0] - abs max y[-1,1] a[-1,1] - abs max y[1,-1] a[1,-1] - abs max y[1,0] a[1,0] '
        '- abs max y[1,1] a[1,1] - abs max y[0,-1] a[0,-1] - abs max y[0,1] a[0,1] - abs max y a - abs max y[-1,-1] z[-1,-1] '
        '- abs y[-1,0] z[-1,0] - abs max y[-1,1] z[-1,1] - abs max y[1,-1] z[1,-1] - abs max y[1,0] z[1,0] - abs max y[1,1] '
        'z[1,1] - abs max y[0,-1] z[0,-1] - abs max y[0,1] z[0,1] - abs max y z - abs max min pmax! x y pmax@ - 0 max y '
        f'pmax@ + {full} min clamp',
        # mode 4
        f'a z max max_np! a z min min_np! y min_np@ - 0 max 2 * {full} min min_np@ + {full} min max_np@ min reg5! max_np@ '
        f'max_np@ y - 0 max 2 * {full} min - 0 max min_np@ max reg3! min_np@ reg5@ = max_np@ reg3@ = or y x reg3@ reg5@ '
        'clamp ?'
    ]
    
    orig = clip
    
    clip = chroma_up(clip, planes)
    clip = clip[0] + core.akarin.Expr([clip, chroma_up(refclip, planes), shift_clip(clip, 1), shift_clip(clip, -1)],
                                      [expr[mode] if i in planes else '' for i in range(num_p)])[1:-1] + clip[-1]
    clip = chroma_down(clip, planes)
    
    if not edges and mode in {1, 2, 3}:
        expr = 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?'
        clip = core.akarin.Expr([clip, orig], [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def Clense(clip: vs.VideoNode, previous: vs.VideoNode | None = None, following: vs.VideoNode | None = None, reduceflicker: bool = False,
           planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'Clense'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if previous is None:
        previous = shift_clip(clip, 1)
    elif isinstance(previous, vs.VideoNode) and clip.format.name == previous.format.name and clip.num_frames == previous.num_frames:
        pass
    else:
        raise TypeError(f'{func_name}: invalid "previous"')
    
    if following is None:
        following = shift_clip(clip, -1)
    elif isinstance(following, vs.VideoNode) and clip.format.name == following.format.name and clip.num_frames == following.num_frames:
        pass
    else:
        raise TypeError(f'{func_name}: invalid "following"')
    
    if not isinstance(reduceflicker, bool):
        raise TypeError(f'{func_name}: invalid "reduceflicker"')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    orig = clip
    
    expr = 'x y z min max y z max min'
    
    clip = clip[0] + core.std.Expr([clip, previous, following], [expr if i in planes else '' for i in range(num_p)])[1:-1] + clip[-1]
    
    if reduceflicker:
        clip = clip[0:2] + core.std.Expr([orig, shift_clip(clip, 1), following], [expr if i in planes else '' for i in range(num_p)])[2:-1] + clip[-1]
    
    return clip

def BackwardClense(clip: vs.VideoNode, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'BackwardClense'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = f'x y z max 2 * z - {full} min min y z min 2 * z - 0 max max'
    
    clip = chroma_up(clip, planes)
    clip = clip[:2] + core.std.Expr([clip, shift_clip(clip, 1), shift_clip(clip, 2)], [expr if i in planes else '' for i in range(num_p)])[2:]
    clip = chroma_down(clip, planes)
    
    return clip

def ForwardClense(clip: vs.VideoNode, planes: int | list[int] | None = None) -> vs.VideoNode:
    
    func_name = 'ForwardClense'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = f'x y z max 2 * z - {full} min min y z min 2 * z - 0 max max'
    
    clip = chroma_up(clip, planes)
    clip = core.std.Expr([clip, shift_clip(clip, -1), shift_clip(clip, -2)], [expr if i in planes else '' for i in range(num_p)])[:-2] + clip[-2:]
    clip = chroma_down(clip, planes)
    
    return clip

def VerticalCleaner(clip: vs.VideoNode, mode: int | list[int] = 1, edges: bool = False) -> vs.VideoNode:
    
    func_name = 'VerticalCleaner'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match mode:
        case int() if 0 <= mode <= 2:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and 0 <= i <= 2 for i in mode):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "mode"')
    
    if not isinstance(edges, bool):
        raise TypeError(f'{func_name}: invalid "edges"')
    
    expr = [
        # mode 0
        '',
        # mode 1
        'x[0,-1] x[0,1] min x max x[0,-1] x[0,1] max min',
        # mode 2
        'x x[0,-1] x[0,1] min x[0,-1] x[0,-2] x[0,-1] - 0 max - 0 max x[0,1] x[0,2] x[0,1] - 0 max - 0 max max min x[0,-1] '
        f'x[0,-2] - 0 max x[0,-1] + {full} min x[0,1] x[0,2] - 0 max x[0,1] + {full} min min x[0,-1] max x[0,1] max clamp'
    ]
    
    orig = clip
    
    planes = [i for i, j in enumerate(mode + [mode[-1]] * (num_p - len(mode))) if j]
    clip = chroma_down(core.akarin.Expr(chroma_up(clip, planes), [expr[i] for i in mode]), planes)
    
    if not edges:
        expr = ['',
                # mode 1
                'Y 0 = Y height 1 - = or y x ?',
                # mode 2
                'Y 1 <= Y height 2 - >= or y x ?']
        
        clip = core.akarin.Expr([clip, orig], [expr[i] for i in mode])
    
    return clip

def Convolution(clip: vs.VideoNode, mode: str | list[int] | list[list[int]] | None = None, saturate: int | None = None,
                total: float | None = None, planes: int | list[int] | None = None) -> vs.VideoNode:
    """
    An unnatural hybrid of std.Convolution, mt_convolution and mt_edge.
    
    All named modes from mt_edge are present. The kernel can also be specified as two flat matrices or a square matrix.
    Unlike std.Convolution, it works correctly with edges.
    The default mode value is boxblur 3x3.
    The default value of saturate is 1.
    The default value of total is the sum of the absolute values of the resulting matrix.
    For named modes, the default values are changed to obtain the desired result, but they can be overridden by specifying them explicitly.
    """
    func_name = 'Convolution'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1 if clip.format.sample_type == vs.INTEGER else 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    match mode:
        case [[int(), *a], [int(), *b]] if all(isinstance(i, int) for i in a + b) and len(mode[0]) % 2 and len(mode[1]) % 2:
            side_h, side_v = len(mode[0]), len(mode[1])
            mode = [j * i for i in mode[1] for j in mode[0]]
        case [int(), *a] if all(isinstance(i, int) for i in a) and (side_v := round(sqrt(len(mode)))) ** 2 == len(mode) and side_v % 2:
            side_h = side_v
        case None:
            side_h = side_v = 3
            mode = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        case 'sobel':
            side_h = side_v = 3
            mode = [0, -1, 0, -1, 0, 1, 0, 1, 0]
            fix = ' abs'
            div = 2
        case 'roberts':
            side_h = side_v = 3
            mode = [0, 0, 0, 0, 2, -1, 0, -1, 0]
            fix = ' abs'
            div = 2
        case 'laplace':
            side_h = side_v = 3
            mode = [1, 1, 1, 1, -8, 1, 1, 1, 1]
            fix = ' abs'
            div = 8
        case 'cartoon':
            side_h = side_v = 3
            mode = [0, 2, -1, 0, -1, 0, 0, 0, 0]
            fix = ' -1 *'
            div = 1
        case 'min/max':
            expr = ('x[-1,-1] x[0,-1] max x[1,-1] max x[-1,0] max x max x[1,0] max x[-1,1] max x[0,1] max x[1,1] max '
                    'x[-1,-1] x[0,-1] min x[1,-1] min x[-1,0] min x min x[1,0] min x[-1,1] min x[0,1] min x[1,1] min -')
            div = 1
        case 'hprewitt':
            mode = [[1, 2, 1, 0, 0, 0, -1, -2, -1], [1, 0, -1, 2, 0, -2, 1, 0, -1]]
            return core.std.Expr([Convolution(clip, i, 0 if saturate is None else saturate, 1.0 if total is None else total, planes) for i in mode],
                                 ['x y max' if i in planes else '' for i in range(num_p)])
        case 'prewitt':
            mode = [[1, 1, 0, 1, 0, -1, 0, -1, -1], [1, 1, 1, 0, 0, 0, -1, -1, -1],
                    [1, 0, -1, 1, 0, -1, 1, 0, -1], [0, -1, -1, 1, 0, -1, 1, 1, 0]]
            return core.std.Expr([Convolution(clip, i, 0 if saturate is None else saturate, 1.0 if total is None else total, planes) for i in mode],
                                 ['x y max z a max max' if i in planes else '' for i in range(num_p)])
        case 'kirsch4':
            mode = [[5, 5, 5, -3, 0, -3, -3, -3, -3], [5, -3, -3, 5, 0, -3, 5, -3, -3],
                    [-3, -3, -3, -3, 0, -3, 5, 5, 5], [-3, -3, 5, -3, 0, 5, -3, -3, 5]]
            return core.std.Expr([Convolution(clip, i, 0 if saturate is None else saturate, 1.0 if total is None else total, planes) for i in mode],
                                 ['x y max z a max max' if i in planes else '' for i in range(num_p)])
        case 'kirsch8':
            mode = [[5, 5, 5, -3, 0, -3, -3, -3, -3], [5, 5, -3, 5, 0, -3, -3, -3, -3],
                    [5, -3, -3, 5, 0, -3, 5, -3, -3], [-3, -3, -3, 5, 0, -3, 5, 5, -3],
                    [-3, -3, -3, -3, 0, -3, 5, 5, 5], [-3, -3, -3, -3, 0, 5, -3, 5, 5],
                    [-3, -3, 5, -3, 0, 5, -3, -3, 5], [-3, 5, 5, -3, 0, 5, -3, -3, -3]]
            return core.std.Expr([Convolution(clip, i, 0 if saturate is None else saturate, 1.0 if total is None else total, planes) for i in mode],
                                 ['x y max z a max max b c max d e max max max' if i in planes else '' for i in range(num_p)])
        case _:
            raise TypeError(f'{func_name}: invalid "mode"')
    
    match saturate:
        case None:
            if 'fix' not in locals():
                fix = ''
        case 0:
            fix = ' abs'
        case 1:
            fix = ''
        case 2:
            fix = ' -1 *'
        case _:
            raise TypeError(f'{func_name}: invalid "saturate"')
    
    match total:
        case None:
            if 'div' not in locals():
                div = sum(abs(i) for i in mode)
        case float():
            div = total
        case _:
            raise TypeError(f'{func_name}: invalid "total"')
    
    if 'expr' in locals():
        expr = f'{expr} {div} /{fix} 0 {full} clamp'
    else:
        expr = (f'{' '.join(f'x[{j - (side_h // 2)},{i - (side_v // 2)}] {mode[i * side_h + j]} *' for i in range(side_v) for j in range(side_h))} '
                f'{'+ ' * (len(mode) - 1)}{div} /{fix} 0 {full} clamp')
    
    clip = chroma_down(core.akarin.Expr(chroma_up(clip, planes), [expr if i in planes else '' for i in range(num_p)]), planes)
    
    return clip

def CrazyPlaneStats(clip: vs.VideoNode, mode: int | list[int] = 0, plane: int = 0, norm: bool = True) -> vs.VideoNode:
    """
    Calculates arithmetic mean, geometric mean, arithmetic-geometric mean, harmonic mean, contraharmonic mean,
    root mean square, root mean cube and median, depending on the mode.
    
    The result is written to the frame properties with the corresponding name.
    """  # noqa: D205
    from scipy import special
    
    func_name = 'CrazyPlaneStats'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    
    if not isinstance(plane, int) or plane < 0 or plane >= num_p:
        raise ValueError(f'{func_name}: invalid "plane"')
    
    if clip.format.sample_type == vs.INTEGER:
        full = (1 << clip.format.bits_per_sample) - 1
        isfloat = False
    else:
        full = 1
        isfloat = True
        norm = False
    
    match mode:
        case int() if 0 <= mode <= 7:
            mode = [mode]
        case list() if mode and all(isinstance(i, int) and 0 <= i <= 7 for i in mode):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "mode"')
    
    if 2 in mode:
        if 0 not in mode:
            mode += [0]
        if 1 not in mode:
            mode += [1]
        mode.sort()
    
    if not isinstance(norm, bool):
        raise TypeError(f'{func_name}: invalid "norm"')
    
    def frame_stats(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        
        fout = f.copy()
        
        matrix = np.asarray(f[plane])
        
        fout.props['maximum'] = np.amax(matrix).astype(np.float64) if isfloat else int(np.amax(matrix))
        fout.props['minimum'] = np.amin(matrix).astype(np.float64) if isfloat else int(np.amin(matrix))
        
        for i in mode:
            match i:
                case 0:
                    avg = avg_a = np.mean(matrix, dtype=np.float64)
                    name = 'arithmetic_mean'
                case 1:
                    avg = avg_g = np.exp(np.mean(np.log(matrix, dtype=np.float64)))
                    name = 'geometric_mean'
                case 2:
                    avg = np.pi * (avg_a + avg_g) / special.ellipk(np.square(avg_a - avg_g) / np.square(avg_a + avg_g)) / 4
                    name = 'arithmetic_geometric_mean'
                case 3:
                    avg = matrix.size / np.sum(np.reciprocal(matrix, dtype=np.float64))
                    name = 'harmonic_mean'
                case 4:
                    avg = np.mean(np.square(matrix, dtype=np.float64 if isfloat else np.uint32), dtype=np.float64) / np.mean(matrix, dtype=np.float64)
                    name = 'contraharmonic_mean'
                case 5:
                    avg = np.sqrt(np.mean(np.square(matrix, dtype=np.float64 if isfloat else np.uint32), dtype=np.float64))
                    name = 'root_mean_square'
                case 6:
                    avg = np.cbrt(np.mean(matrix.astype(np.float64 if isfloat else np.uint64) ** 3, dtype=np.float64))
                    name = 'root_mean_cube'
                case 7:
                    avg = np.median(matrix)
                    name = 'median'
            
            if norm:
                avg /= full
            
            fout.props[name] = avg - 0.5 if isfloat and plane else avg
        
        return fout
    
    clip = chroma_up(clip, [plane])
    clip = chroma_down(core.std.ModifyFrame(clip=clip, clips=clip, selector=frame_stats), [plane])
    
    return clip

def out_of_range_search(clip: vs.VideoNode, lower: int | None = None, upper: int | None = None, output: str | None = None,
                        planes: int | list[int] | None = None) -> vs.VideoNode:
    """Searches for pixel values outside the specified range. The found values are written to a text file."""
    func_name = 'out_of_range_search'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_f = clip.num_frames
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor - 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    check = 0
    
    match lower:
        case None:
            pass
        case int() if 0 <= lower <= 255:
            lower *= factor
            check += 1
        case _:
            raise TypeError(f'{func_name}: invalid "lower"')
    
    match upper:
        case None:
            pass
        case int() if 0 <= upper <= 255:
            upper *= factor
            check += 2
        case _:
            raise TypeError(f'{func_name}: invalid "upper"')
    
    if not check:
        raise ValueError(f'{func_name}: "lower" and "upper" cannot both be None')
    
    match output:
        case None:
            output = (f'out_of_range_{f'lower({lower})_' if lower is not None else ''}'
                      f'{f'upper({upper})_' if upper is not None else ''}planes({'_'.join(f'{i}' for i in planes)}).txt')
        case str():
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "output"')
    
    out_of_range = [None] * num_f * len(planes)
    counter = np.full(num_f, np.False_, dtype=np.bool_)
    
    def get_search(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        
        nonlocal out_of_range, counter
        
        for i in planes:
            matrix = np.asarray(f[i])
            
            match check:
                case 1:
                    temp = np.where(matrix < lower)
                case 2:
                    temp = np.where(matrix > upper)
                case 3:
                    temp = np.where((matrix < lower) | (matrix > upper))
            
            if temp[0].size:
                out_of_range[n * len(planes) + i - min(planes)] = [i, n, temp, matrix[temp]]
            
        counter[n] = np.True_
        
        if np.all(counter):
            
            dig = max(len(str(num_f)), 5)
            w = len(str(clip.width))
            h = len(str(clip.height))
            out = max(len(str(full)), 3)
            
            res = [f'{x[1]:>{dig}} {j:>{w}} {i:>{h}} {k:>{out}} {x[0]:>5}\n' for x in out_of_range if x is not None
                   for i, j, k in zip(*x[2], x[3])]
            
            if res:
                with open(output, 'w') as file:
                    file.write(f'{'frame':>{dig}} {'x':>{w}} {'y':>{h}} {'out':>{out}} plane\n')
                    file.writelines(res)
            else:
                raise ValueError(f'{func_name}: there is no result, check the settings')
        
        return clip
    
    clip = core.std.FrameEval(clip, partial(get_search, clip=clip), prop_src=clip, clip_src=clip)
    
    return clip

def rescaler(clip: vs.VideoNode, dx: float | None = None, dy: float | None = None, kernel: str = 'bilinear',
             mode: int = 5, upscaler: Callable | None = None, ratio: float = 1.0,
             **descale_args: Any) -> vs.VideoNode:
    
    func_name = 'rescaler'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.FLOAT:
        raise TypeError(f'{func_name}: integer sample type is not supported')
    
    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    w = clip.width
    h = clip.height
    blunt_w = 1 << clip.format.subsampling_w
    blunt_h = 1 << clip.format.subsampling_h
    crop_keys = {'src_left', 'src_top', 'src_width', 'src_height'}
    
    if descale_args and mode & 1 and (x := crop_keys & set(descale_args.keys())):
        raise ValueError(f'{func_name}: Unsupported key(s) {x} in descale_args')
    
    if not isinstance(mode, int):
        raise TypeError(f'{func_name}: invalid mode')
    
    if mode & 2:
        # https://web.archive.org/web/20231123073420/https://anibin.blogspot.com/2014/01/blog-post_3155.html
        if dx is None:
            raise TypeError(f'{func_name}: invalid "dx" for studio resolution mode')
        if dy is None:
            raise TypeError(f'{func_name}: invalid "dy" for studio resolution mode')
        if mode & 1:
            match w, h:
                case 1920, 1080:
                    up_w = 1088 * 16 / 9
                    dx = dx - (up_w - 1920) * dx / up_w
                    dy = dy - 8 * dy / 1088
                case _:
                    raise ValueError(f'{func_name}: Unsupported resolution for studio resolution mode')
        else:
            raise ValueError(f'{func_name}: Fractional operation mode is required for studio resolution mode')
    
    if kernel not in {'bilinear', 'bicubic', 'lanczos', 'spline16', 'spline36', 'spline64', 'point'}:
        raise ValueError(f'{func_name}: invalid "kernel": {kernel}')
    
    match dx, dy, mode & 1:
        case None, None, 1:
            dy = h * 2 // 3
            descale_args['src_width'] = w * dy * ratio / h
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
        case None, None, 0:
            dy = h * 2 // 3
            dx = round(w * dy * ratio / h)
        case None, int(), 1:
            descale_args['src_width'] = w * dy * ratio / h
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
        case None, int(), 0:
            dx = round(w * dy * ratio / h)
        case None, float(), 1:
            descale_args['src_width'] = w * dy * ratio / h
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
            descale_args['src_height'] = dy
            dy = ceil(descale_args['src_height'] / blunt_h) * blunt_h
            descale_args['src_top'] = (dy - descale_args['src_height']) / 2
        case None, float(), 0:
            dx = round(w * dy * ratio / h)
            dy = round(dy)
        case int(), None, 1:
            descale_args['src_height'] = h * dx * ratio / w
            dy = ceil(descale_args['src_height'] / blunt_h) * blunt_h
            descale_args['src_top'] = (dy - descale_args['src_height']) / 2
        case int(), None, 0:
            dy = round(h * dx * ratio / w)
        case int(), int(), _:
            pass
        case int(), float(), 1:
            descale_args['src_height'] = dy
            dy = ceil(descale_args['src_height'] / blunt_h) * blunt_h
            descale_args['src_top'] = (dy - descale_args['src_height']) / 2
        case int(), float(), 0:
            dy = round(dy)
        case float(), None, 1:
            descale_args['src_width'] = dx
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
            descale_args['src_height'] = h * dx * ratio / w
            dy = ceil(descale_args['src_height'] / blunt_h) * blunt_h
            descale_args['src_top'] = (dy - descale_args['src_height']) / 2
        case float(), None, 0:
            dy = round(h * dx * ratio / w)
            dx = round(dx)
        case float(), int(), 1:
            descale_args['src_width'] = dx
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
        case float(), int(), 0:
            dx = round(dx)
        case float(), float(), 1:
            descale_args['src_width'] = dx
            dx = ceil(descale_args['src_width'] / blunt_w) * blunt_w
            descale_args['src_left'] = (dx - descale_args['src_width']) / 2
            descale_args['src_height'] = dy
            dy = ceil(descale_args['src_height'] / blunt_h) * blunt_h
            descale_args['src_top'] = (dy - descale_args['src_height']) / 2
        case float(), float(), 0:
            dx = round(dx)
            dy = round(dy)
        case None | int() | float(), _:
            raise TypeError(f'{func_name}: invalid "dy"')
        case _:
            raise TypeError(f'{func_name}: invalid "dx"')
    
    clip = getattr(core.descale, f'De{kernel}')(clip, dx, dy, **descale_args)
    
    match upscaler, mode & 12:
        case None, 0:
            clip = getattr(core.descale, kernel.capitalize())(clip, w, h, **descale_args)
        case None, 4:
            resize_keys = dict(src_left='src_left', src_top='src_top', src_width='src_width', src_height='src_height',
                               b='filter_param_a', c='filter_param_b', taps='filter_param_a')
            resize_args = {resize_keys[key]: value for key, value in descale_args.items() if key in resize_keys}
            clip = getattr(core.resize, kernel.capitalize())(clip, w, h, **resize_args)
        case None, 8:
            fmtc_keys = dict(src_left='sx', src_top='sy', src_width='sw', src_height='sh',
                             b='a1', c='a2', taps='taps')
            fmtc_args = {fmtc_keys[key]: value for key, value in descale_args.items() if key in fmtc_keys}
            clip = core.fmtc.resample(clip, w, h, kernel=kernel, **fmtc_args)
        case Callable(), 8:
            fmtc_keys = dict(src_left='sx', src_top='sy', src_width='sw', src_height='sh')
            fmtc_args = {fmtc_keys[key]: value for key, value in descale_args.items() if key in fmtc_keys}
            clip = upscaler(clip, w, h, **fmtc_args)
        case Callable(), _:
            clip = upscaler(clip, w, h, **{key: value for key, value in descale_args.items() if key in crop_keys})
        case _:
            raise TypeError(f'{func_name}: invalid "upscaler"')
    
    return clip

def SCDetect(clip: vs.VideoNode, thr: float = 0.1, luma_only: bool = False) -> vs.VideoNode:
    
    func_name = 'SCDetect'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family == vs.YUV:
        pass
    elif clip.format.color_family == vs.GRAY:
        luma_only = True
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    if luma_only:
        diff = CrazyPlaneStats(core.std.Expr([clip, shift_clip(clip, -1)], ['x y - abs'] + [''] * (num_p - 1)))
        clip = core.akarin.PropExpr([clip, diff], lambda: dict(_SceneChangeNext=f'y.arithmetic_mean {thr * factor / (256 * factor - 1)} > 1 0 ?'))
        clip = core.akarin.PropExpr([clip, shift_clip(clip, 1)], lambda: dict(_SceneChangePrev='y._SceneChangeNext'))
    else:
        diff = core.std.Expr([clip, shift_clip(clip, -1)], 'x y - abs')
        diffs = [CrazyPlaneStats(i) for i in core.std.SplitPlanes(diff)]
        clip = core.akarin.PropExpr([clip, *diffs], lambda: dict(_SceneChangeNext=f'y.arithmetic_mean z.arithmetic_mean a.arithmetic_mean max max {thr * factor / (256 * factor - 1)} > 1 0 ?'))
        clip = core.akarin.PropExpr([clip, shift_clip(clip, 1)], lambda: dict(_SceneChangePrev='y._SceneChangeNext'))
    
    return clip

def getnative(clip: vs.VideoNode, dx: float | list[float] | None = None, dy: float | list[float] | None = None,
              frames: int | list[int | None] | None = None, kernel: str | list[str] = 'bilinear', sigma: int = 0,
              mark: bool = False, output: str | None = None, thr: float = 0.015, crop: int = 5, mean: int = -1,
              yscale: str = 'log', interim: bool = False, figsize: tuple[int, int] = (16, 9),
              layout: str | None = 'tight', **descale_args: Any) -> vs.VideoNode:
    """
    Предупреждение: не смотря на то, что клип представлен как последовательность и имеет те же методы,
    фактически он располагается на жёстком диске и в оперативную память кэшируется лишь его малая часть.
    Поэтому при использовании срезов с большим значением шага неизбежно СИЛЬНОЕ падение производительности.
    """  # noqa: D205
    import gc
    
    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelextrema
    
    func_name = 'getnative'
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name} the clip must be of the vs.VideoNode type')
    
    if clip.format.sample_type != vs.FLOAT:
        raise TypeError(f'{func_name}: integer sample type is not supported')
    
    if clip.format.color_family != vs.GRAY:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    match frames:
        case None if clip.num_frames == 1:
            frames = 0
        case None:
            frames = [0, clip.num_frames]
        case int() | [int(), int()] | [int(), int(), int()]:
            pass
        case [int() | None, int() | None] | [int() | None, int() | None, int() | None]:
            defaults = [0, clip.num_frames, 1]
            frames = [defaults[i] if j is None else j for i, j in enumerate(frames)]
        case _:
            raise TypeError(f'{func_name}: invalid "frames"')
    
    match kernel:
        case 'all' if not any(isinstance(i, list) for i in descale_args.values()):
            kernel = ['bilinear', 'bicubic', 'bicubic', 'bicubic', 'bicubic', 'bicubic', 'bicubic', 'bicubic',
                      'bicubic', 'lanczos', 'lanczos', 'lanczos', 'lanczos', 'spline16', 'spline36', 'spline64']
            descale_args['b'] = [None, 1/3, 0.5, 0, 0, 1, 0, 0.2, 0.5]
            descale_args['c'] = [None, 1/3, 0, 0.5, 0.75, 0, 1, 0.5, 0.5]
            descale_args['taps'] = [None, None, None, None, None, None, None, None, None, 2, 3, 4, 5]
        case str() if any(isinstance(i, list) for i in descale_args.values()):
            kernel = [kernel] * max(len(i) for i in descale_args.values() if isinstance(i, list))
        case str():
            pass
        case list() if (all(isinstance(i, str) for i in kernel) and (not descale_args or
                        len(kernel) >= max(len(i) if isinstance(i, list) else 1 for i in descale_args.values()))):
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "kernel" or "descale_args"')
    
    if not isinstance(sigma, int) or sigma < 0:
        raise TypeError(f'{func_name}: invalid "sigma"')
    
    if not isinstance(interim, bool):
        raise TypeError(f'{func_name}: invalid "interim"')
    
    if layout not in {'tight', 'constrained', 'compressed', 'none', None}:
        raise TypeError(f'{func_name}: invalid "layout"')
    
    match figsize:
        case (int(), int()):
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "figsize"')
    
    match dx, dy, kernel, frames:
        case None | int() | float(), None | int() | float(), list(), int():
            frange = kernel.copy()
            clip = clip[frames] * len(frange)
            descale_args = {key: value + [None] * (len(frange) - len(value)) if isinstance(value, list)
                            else [value] * len(frange) for key, value in descale_args.items()}
            descale_args = [{key: value[i] for key, value in descale_args.items() if value[i] is not None}
                            for i in range(len(frange))]
            resc = core.std.FrameEval(clip, lambda n, clip=clip: rescaler(clip, dx, dy, frange[n], **descale_args[n]),
                                      clip_src=clip)
            param = 'kernel'
        case None | int() | float(), [int() | float(), int() | float(), int() | float()], str(), int():
            frange = np.arange(*dy, dtype=np.float64)
            clip = clip[frames] * len(frange)
            resc = core.std.FrameEval(clip, lambda n, clip=clip: rescaler(clip, dx, frange[n], kernel, **descale_args),
                                      clip_src=clip)
            param = 'dy'
        case [int() | float(), int() | float(), int() | float()], None | int() | float(), str(), int():
            frange = np.arange(*dx, dtype=np.float64)
            clip = clip[frames] * len(frange)
            resc = core.std.FrameEval(clip, lambda n, clip=clip: rescaler(clip, frange[n], dy, kernel, **descale_args),
                                      clip_src=clip)
            param = 'dx'
        case None | int() | float(), None | int() | float(), str(), list():
            frange = np.arange(*frames, dtype=np.int_)
            clip = clip[slice(*frames)]
            resc = rescaler(clip, dx, dy, kernel, **descale_args)
            param = 'frame'
        case None | int() | float(), [int() | float(), int() | float()], str(), list():
            frange = np.arange(*frames, dtype=np.int_)
            clip = clip[slice(*frames)]
            resc = (rescaler(clip, dx, dy[0], kernel, **descale_args) +
                    rescaler(clip, dx, dy[1], kernel, **descale_args))
            clip *= 2
            param = 'frame_dy'
        case [int() | float(), int() | float()], None | int() | float(), str(), list():
            frange = np.arange(*frames, dtype=np.int_)
            clip = clip[slice(*frames)]
            resc = (rescaler(clip, dx[0], dy, kernel, **descale_args) +
                    rescaler(clip, dx[1], dy, kernel, **descale_args))
            clip *= 2
            param = 'frame_dx'
        case [int() | float(), int() | float()], [int() | float(), int() | float()], str(), list():
            frange = np.arange(*frames, dtype=np.int_)
            clip = clip[slice(*frames)]
            resc = (rescaler(clip, dx[0], dy[0], kernel, **descale_args) +
                    rescaler(clip, dx[1], dy[1], kernel, **descale_args))
            clip *= 2
            param = 'frame_dx_dy'
        case None | int() | float(), None | int() | float(), list(), list():
            frange = kernel.copy()
            clip = clip[slice(*frames)]
            descale_args = {key: value + [None] * (len(frange) - len(value)) if isinstance(value, list)
                            else [value] * len(frange) for key, value in descale_args.items()}
            descale_args = [{key: value[i] for key, value in descale_args.items() if value[i] is not None}
                            for i in range(len(frange))]
            resc = core.std.Splice([rescaler(clip, dx, dy, i, **j) for i, j in zip(frange, descale_args)])
            clip *= len(frange)
            param = 'total_kernel'
        case None | int() | float(), [int() | float(), int() | float(), int() | float()], str(), list():
            frange = np.arange(*dy, dtype=np.float64)
            clip = clip[slice(*frames)]
            resc = core.std.Splice([rescaler(clip, dx, i, kernel, **descale_args) for i in frange])
            clip *= len(frange)
            param = 'total_dy'
        case [int() | float(), int() | float(), int() | float()], None | int() | float(), str(), list():
            frange = np.arange(*dx, dtype=np.float64)
            clip = clip[slice(*frames)]
            resc = core.std.Splice([rescaler(clip, i, dy, kernel, **descale_args) for i in frange])
            clip *= len(frange)
            param = 'total_dx'
        case _:
            raise TypeError(f'{func_name}: unsupported combination of parameters')
    
    match output:
        case None:
            output = f'pass_{param}_frame(s)_{frames}_{dx}x{dy}.txt'
        case str() if output.split('.')[-1] == 'txt':
            pass
        case _:
            raise TypeError(f'{func_name}: invalid "output"')
    
    clip = core.akarin.Expr([clip, resc], f'x y - abs var! var@ {thr} > var@ 0 ?')
    
    if crop:
        clip = core.std.Crop(clip, crop, crop, crop, crop)
    
    clip = core.std.PlaneStats(clip) if mean == -1 else CrazyPlaneStats(clip, mean)
    
    result = np.zeros(clip.num_frames, dtype=np.float64)
    counter = np.full(clip.num_frames, np.False_, dtype=np.bool_)
    
    means = ['arithmetic_mean', 'geometric_mean', 'arithmetic_geometric_mean', 'harmonic_mean', 'contraharmonic_mean',
             'root_mean_square', 'root_mean_cube', 'median', 'PlaneStatsAverage']
    
    class GetPlot:
        def __enter__(self) -> Self:
            self.fig, self.ax = plt.subplots(figsize=figsize, layout=layout)
            return self
        
        def plot(self, sfrange: list[str], frange: list[str] | np.ndarray, result: np.ndarray, output: str,
                 param: str, y_lim: tuple[np.float64, np.float64] | None = None) -> None:
            
            dig = max(max(len(i) for i in sfrange), len(param))
            
            min_index = argrelextrema(result, np.less)[0]
            min_label = [' local min' if i in min_index else '' for i in range(len(frange))]
            
            if param in {'frame_dx', 'frame_dy', 'frame_dx_dy'}:
                res = [f'{i:>{dig}} {j:20.2f}{k}\n' for i, j, k in zip(sfrange, result, min_label)]
            else:
                res = [f'{i:>{dig}} {j:.20f}{k}\n' for i, j, k in zip(sfrange, result, min_label)]
            
            if res:
                p = Path(output)
                p.parent.mkdir(exist_ok=True)
                with p.open('w') as file:
                    file.write(f'{param:<{dig}} abs diff\n')
                    file.writelines(res)
            else:
                raise ValueError(f'{func_name}: there is no result, check the settings')
            
            self.ax.plot(frange, result)
            self.ax.set(yscale=yscale, xlabel=param, ylabel='absolute normalized difference')
            if y_lim is not None:
                self.ax.set_ylim(*y_lim)
            self.ax.grid()
            
            if mark:
                if param in {'kernel', 'total_kernel'}:
                    self.ax.plot(min_index, result[min_index], marker='x', c='k', ls='')
                    for i, j in zip(min_index, result[min_index]):
                        self.ax.annotate(f'{j:.2e}', (i, j), textcoords='offset points',
                                         xytext=(6, 12), ha='right', va='bottom', rotation=90,
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                else:
                    self.ax.plot(frange[min_index], result[min_index], marker='x', c='k', ls='')
                    for i, j, k in zip(frange[min_index], result[min_index], np.array(sfrange)[min_index]):
                        self.ax.annotate(k, (i, j), textcoords='offset points',
                                         xytext=(6, 12), ha='right', va='bottom', rotation=90,
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            self.fig.savefig(p.with_suffix('.png'), format='png')
            self.ax.clear()
        
        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
            plt.close(self.fig)
    
    def get_native(n: int, f: vs.VideoFrame, clip: vs.VideoNode, frange: list[str] | np.ndarray) -> vs.VideoNode:
        
        nonlocal result, counter
        result[n] = f.props[means[mean]]
        counter[n] = np.True_
        
        if np.all(counter):
            result[result < 1e-9] = 1e-9
            
            match frange[0]:
                case str():
                    if len(frange) != len(x := set(frange)):
                        temp = {i: [j for j, k in enumerate(frange) if i == k] for i in x}
                        frange = [f'{j}#{temp[j].index(i)}' if len(temp[j]) > 1 else j for i, j in enumerate(frange)]
                    sfrange = frange
                case np.int_():
                    if param != 'frame':
                        result = np.divide(*result.reshape(2, -1))
                    sfrange = [str(i) for i in frange]
                case np.float64():
                    tale_0 = str(frange[0]).split('.')[1]
                    tale_1 = str(frange[1]).split('.')[1]
                    if int(tale_0) or int(tale_1):
                        digits = max(len(tale_0), len(tale_1))
                        sfrange = [f'{i:.{digits}f}' for i in frange]
                    else:
                        sfrange = [str(int(i)) for i in frange]
            
            with GetPlot() as plot:
                if param in {'total_kernel', 'total_dy', 'total_dx'}:
                    result = result.reshape(len(frange), -1).T
                    if sigma:
                        result = gaussian_filter(result, sigma, axes=1)
                    if interim:
                        import sys
                        y_lim = (np.amin(result), np.amax(result))
                        for i, j, k in zip(result, range(result.shape[0]), range(*frames)):
                            print(f'Frame: {j}/{result.shape[0]} - "interim" pass{' ':<20}', end='\r', file=sys.stderr)
                            plot.plot(sfrange, frange, i, f'{output[:-4]}/frame_{k}.txt', param[6:], y_lim)
                    result = np.exp(np.mean(np.log(result), axis=0))
                elif sigma:
                    result = gaussian_filter(result, sigma)
                
                plot.plot(sfrange, frange, result, output, param)
            
            gc.collect()
        
        return clip
    
    clip = core.std.FrameEval(clip, partial(get_native, clip=clip, frange=frange), prop_src=clip, clip_src=clip)
    
    return clip

# Подумать насчёт деления на 255 в float. Возможно стоит сделать 256.
# Обязательны к тщательной проверке на float: Blur, UnsharpMask, RemoveGrain, MinBlur, Repair, sbr, Clamp.
# добавить поддержку float в average_fields
# проверить как ведёт себя PlaneStats на хрома-флоатах, сранить с CrazyPlaneStats
# search_field_diffs - убрать нахрен нормализацию и добавить PlaneStats по-умолчанию
