'''
All functions support the following formats: GRAY and YUV 8 - 16 bit integer. Floating point sample type is not supported.

Functions:
    autotap3
    Lanczosplus
    bion_dehalo
    fix_border
    MaskDetail
    degrain_n
    Destripe
    daa
    average_fields
    znedi3aas
    dehalo_mask
    tp7_deband_mask
    DeHalo_alpha
    FineDehalo
    FineDehalo2
    InsaneAA
    upscaler
    custom_mask
    diff_mask
    apply_range
    titles_mask
    after_mask
    search_field_diffs
    CombMask2
    mt_CombMask
    mt_binarize
    delcomb
    vinverse
    vinverse2
    sbr
    sbrV
    avs_Blur
    avs_Sharpen
    mt_clamp
    MinBlur
    Dither_Luma_Rebuild
    mt_expand_multi
    mt_inpand_multi
    avs_TemporalSoften
    UnsharpMask
    diff_tfm
    diff_transfer
    shift_clip
    ovr_comparator
    RemoveGrain
    Repair
    TemporalRepair
    Clense
    BackwardClense
    ForwardClense
    VerticalCleaner
    Convolution
'''

from vapoursynth import core, GRAY, YUV, VideoNode, VideoFrame, INTEGER
from muvsfunc import rescale
from typing import Any, Seq
from math import sqrt
from functools import partial
from inspect import signature
from collections.abc import Callable
import re

def autotap3(clip: VideoNode, dx: int | None = None, dy: int | None = None, mtaps3: int = 1, thresh: int = 256, **crop_args: float) -> VideoNode:
    '''
    Lanczos-based resize from "*.mp4 guy", ported from AviSynth version with minor modifications.
    In comparison with the original, processing accuracy has been doubled, support for 8-16 bit depth
    and crop parameters has been added, and dead code has been removed.
    
    dx and dy are the desired resolution. The other parameters are not documented in any way and are selected using the poke method.
    Cropping options are added as **kwargs. The key names are the same as in VapourSynth-resize.
    '''
    
    func_name = 'autotap3'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w * 2
    
    if dy is None:
        dy = h * 2
    
    if dx == w and dy == h:
        return core.resize.Spline36(clip, **crop_args)
    
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
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
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
    
    m1 = core.akarin.Expr([clip, core.resize.Lanczos(t1, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m2 = core.akarin.Expr([clip, core.resize.Lanczos(t2, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m3 = core.akarin.Expr([clip, core.resize.Lanczos(t3, w, h, filter_param_a=1, **back_args)], 'x y - abs')
    m4 = core.akarin.Expr([clip, core.resize.Lanczos(t4, w, h, filter_param_a=2, **back_args)], 'x y - abs')
    m5 = core.akarin.Expr([clip, core.resize.Lanczos(t5, w, h, filter_param_a=2, **back_args)], 'x y - abs')
    m6 = core.akarin.Expr([clip, core.resize.Lanczos(t6, w, h, filter_param_a=3, **back_args)], 'x y - abs')
    m7 = core.akarin.Expr([clip, core.resize.Lanczos(t7, w, h, filter_param_a=6, **back_args)], 'x y - abs')
    
    expr = f'x y - {thresh} *'
    
    cp1 = core.std.MaskedMerge(avs_Blur(t1, 1.42), t2, core.akarin.Expr([m1, m2], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    m100 = core.akarin.Expr([clip, core.resize.Bilinear(cp1, w, h, **back_args)], 'x y - abs')
    cp2 = core.std.MaskedMerge(cp1, t3, core.akarin.Expr([m100, m3], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    m101 = core.akarin.Expr([clip, core.resize.Bilinear(cp2, w, h, **back_args)], 'x y - abs')
    cp3 = core.std.MaskedMerge(cp2, t4, core.akarin.Expr([m101, m4], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    m102 = core.akarin.Expr([clip, core.resize.Bilinear(cp3, w, h, **back_args)], 'x y - abs')
    cp4 = core.std.MaskedMerge(cp3, t5, core.akarin.Expr([m102, m5], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    m103 = core.akarin.Expr([clip, core.resize.Bilinear(cp4, w, h, **back_args)], 'x y - abs')
    cp5 = core.std.MaskedMerge(cp4, t6, core.akarin.Expr([m103, m6], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    m104 = core.akarin.Expr([clip, core.resize.Bilinear(cp5, w, h, **back_args)], 'x y - abs')
    clip = core.std.MaskedMerge(cp5, t7, core.akarin.Expr([m104, m7], expr).resize.Lanczos(dx, dy, filter_param_a=mtaps3, **crop_args))
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy, **crop_args)], list(range(orig.format.num_planes)), space)
    
    return clip

def Lanczosplus(clip: VideoNode, dx: int | None = None, dy: int | None = None, thresh: int = 0, thresh2: int | None = None,
                 athresh: int = 256, sharp1: float = 1, sharp2: float = 4, blur1: float = 0.33, blur2: float = 1.25,
                 mtaps1: int = 1, mtaps2: int = 1, ttaps: int = 1, ltaps: int = 1, preblur: bool = False, depth: int = 2,
                 wthresh: int = 230, wblur: int = 2, mtaps3: int = 1) -> VideoNode:
    '''
    An upscaler based on Lanczos and AWarpSharp from "*.mp4 guy", ported from AviSynth version with minor modifications.
    In comparison with the original, the mathematics for non-multiple resolutions has been improved, support for 8-16 bit depth
    has been added, dead code and unnecessary calculations have been removed.
    All dependent parameters have been recalculated from AWarpSharp to AWarpSharp2.
    It comes with autotap3, ported just for completion.
    
    dx and dy are the desired resolution. The other parameters are not documented in any way and are selected using the poke method.
    '''
    
    func_name = 'Lanczosplus'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
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
    thresh *= 1 << clip.format.bits_per_sample - 8
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    fd1 = core.resize.Lanczos(clip, dx, dy, filter_param_a=mtaps1)
    fre1 = core.resize.Lanczos(fd1, w, h, filter_param_a=mtaps1)
    fre2 = autotap3(fre1, x := max(w // 16 * 8, 144), y := max(h // 16 * 8, 144), mtaps3, athresh)
    fre2 = autotap3(fre2, w, h, mtaps3, athresh)
    m1 = core.akarin.Expr([fre1, clip], f'x y - abs {thresh} - {thresh2} *')
    m2 = core.resize.Lanczos(core.resize.Lanczos(core.frfun7.Frfun7(m1, 2.01, 256, 256), x, y, filter_param_a=ttaps), dx, dy, filter_param_a=ttaps)
    
    d = core.std.MaskedMerge(clip, fre2, m1) if preblur else clip
    d2 = autotap3(d, dx, dy, mtaps3, athresh)
    d3 = core.resize.Lanczos(core.resize.Lanczos(d, w, h, filter_param_a=ttaps), dx, dy, filter_param_a=ttaps)
    d4 = core.std.MaskedMerge(core.akarin.Expr([d2, d3],  f'x y - {sharp1} * x +'), core.akarin.Expr([d2, d3],  f'y x - {blur1} * x +'), m2)
    d5 = autotap3(d4, w, h, mtaps3, athresh)
    
    e = autotap3(core.std.MaskedMerge(d5, clip, m1), dx, dy, mtaps3, athresh)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    e = core.warp.AWarpSharp2(e, thresh=wthresh, blur=wblur, depth=depth)
    
    fd12 = core.resize.Lanczos(e, dx ** 2 // w // 16 * 16, dy ** 2 // h // 16 * 16, filter_param_a=mtaps2)
    fre12 = core.resize.Lanczos(fd12, dx, dy, filter_param_a=mtaps2)
    m12 = core.akarin.Expr([fre12, e], f'x y - abs {thresh} - {thresh2} *')
    m12 = core.resize.Lanczos(m12, max(dx // 16 * 8, 144), max(dy // 16 * 8, 144), filter_param_a=mtaps2).resize.Lanczos(dx, dy, filter_param_a=mtaps2)
    
    e2 = core.resize.Lanczos(core.resize.Lanczos(e, w, h, filter_param_a=ltaps), dx, dy, filter_param_a=ltaps)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    e2 = core.warp.AWarpSharp2(e2, thresh=wthresh, blur=wblur, depth=depth)
    
    e3 = core.std.MaskedMerge(core.akarin.Expr([e, e2], f'y x - {blur2} * x +'), core.akarin.Expr([e, e2], f'x y - {sharp2} * x +'), m12)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    e3 = core.warp.AWarpSharp2(e3, thresh=wthresh, blur=wblur, depth=depth)
    
    clip = core.std.MaskedMerge(d4, e3, m2)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy)], list(range(orig.format.num_planes)), space)
    
    return clip

def bion_dehalo(clip: VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1, m: bool = False) -> VideoNode:
    '''
    Dehalo by bion, ported from AviSynth version with minor additions.
    
    Args:
        mode: Repair mode from dehaloed clip.
            1, 5, 11 - the weakest, artifacts will not cause.
            2, 3, 4 - bad modes, eat innocent parts, can't be used.
            10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
            14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.
        rep: use Repair to clamp result clip or not.
        rg: use RemoveGrain and Repair to merge with blurred clip or not.
        mask: the mask to merge clip and blurred clip.
            3 - the most accurate.
            4 - the roughest.
            1 and 2 - somewhere in the middle.
        m: show the mask instead of the clip or not.
    '''
    
    func_name = 'bion_dehalo'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    e1 = core.akarin.Expr(Convolution(clip, 'min-max'), f'x {4 * factor} - 4 *')
    e2 = core.std.Maximum(e1).std.Maximum()
    e2 = core.std.Merge(e2, core.std.Maximum(e2)).std.Inflate()
    e3 = core.akarin.Expr([core.std.Merge(e2, core.std.Maximum(e2)), core.std.Deflate(e1)], 'x y 1.2 * -').std.Inflate()
    
    m1 = core.akarin.Expr([clip, UnsharpMask(clip, 40, 2, 0)], f'x y - {factor} - 128 *').std.Maximum().std.Inflate()
    m2 = core.std.Maximum(m1).std.Maximum()
    m3 = RemoveGrain(core.akarin.Expr([m1, m2], 'y x -'), 21).std.Maximum()
    
    n1 = core.akarin.Expr([clip, UnsharpMask(clip, 40, 2, 0, 'gauss')], f'x y - {factor} - 128 *').std.Maximum().std.Inflate()
    n2 = core.std.Maximum(n1).std.Maximum()
    n3 = RemoveGrain(core.akarin.Expr([n1, n2], 'y x -'), 21).std.Maximum()
    
    match mask:
        case 1:
            pass
        case 2:
            e3 = m3
        case 3:
            e3 = core.akarin.Expr([e3, m3], 'x y min')
        case 4:
            e3 = core.akarin.Expr([e3, m3], 'x y max')
        case 5:
            e3 = n3
        case 6:
            e3 = core.akarin.Expr([e3, n3], 'x y min')
        case _:
            raise ValueError(f'{func_name}: Please use 1...6 mask value')
    
    blurr = RemoveGrain(RemoveGrain(MinBlur(clip, 1), 11), 11)
    
    if rg:
        dh1 = core.std.MaskedMerge(Repair(clip, RemoveGrain(clip, 21), 1), blurr, e3)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, e3)
    
    dh1D = core.std.MakeDiff(clip, dh1)
    tmp = sbr(dh1)
    med2D = core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2))
    DD  = core.akarin.Expr([dh1D, med2D], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs 2 * < x y {half} - 2 * {half} + ? ?')
    dh2 = core.std.MergeDiff(dh1, DD)
    
    clip = mt_clamp(clip, Repair(clip, dh2, mode) if rep else dh2, clip, 0, 20)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if m:
        clip = core.resize.Point(e3, format=orig.format.id) if space == YUV else e3
    
    return clip

def fix_border(clip: VideoNode, *args: str | list[str | int | None]) -> VideoNode:
    '''
    A simple functions for fix brightness artifacts at the borders of the frame.
    
    All values are set as positional list arguments. The list have the following format:
    [axis, target, donor, limit, curve, plane]. Only axis is mandatory.
    
    Args:
        axis: can take the values "x" or "y" for columns and rows, respectively.
        target: the target column/row, it is counted from the upper left edge of the screen, by default 0.
        donor: the donor column/row, by default "None" (is calculated automatically as one closer to the center of the frame).
        limit: by default 0, without restrictions, positive values prohibit the darkening of target rows/columns
            and limit the maximum lightening, negative values - on the contrary, it's set in 8-bit notation.
        curve: target correction curve, by default 1, 0 - subtraction and addition, -1 and 1 - division and multiplication,
            -2 and 2 - logarithm and exponentiation, -3 and 3 - nth root and exponentiation.
        plane: by default 0.
    
    Example:
        clip = fix_border(clip, ['x', 0, 1, 50], ['x', 1919, 1918, 50], ['y', 0, 1, 50], ['y', 1079, 1078, 50])
    '''
    
    func_name = 'fix_border'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        num_p = clip.format.num_planes
        clips = core.std.SplitPlanes(clip)
    elif space == GRAY:
        clips = [clip]
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    def axis_x(clip: VideoNode, target: int, donor: int | None, limit: int, curve: int) -> VideoNode:
        
        w = clip.width
        
        if donor is None:
            donor = target + 1 if target < w // 2 else target - 1
        
        target_line = core.std.Crop(clip, target, w - target - 1, 0, 0)
        donor_line = core.std.Crop(clip, donor, w - donor - 1, 0, 0)
        
        fix_line = correction(target_line, donor_line, limit, curve)
        
        if target == 0:
            clip = core.std.StackHorizontal([fix_line, core.std.Crop(clip, 1, 0, 0, 0)])
        elif target == w - 1:
            clip = core.std.StackHorizontal([core.std.Crop(clip, 0, 1, 0, 0), fix_line])
        else:
            clip = core.std.StackHorizontal([core.std.Crop(clip, 0, w - target, 0, 0), fix_line, core.std.Crop(clip, target + 1, 0, 0, 0)])
        
        return clip
    
    def axis_y(clip: VideoNode, target: int, donor: int | None, limit: int, curve: int) -> VideoNode:
        
        h = clip.height
        
        if donor is None:
            donor = target + 1 if target < h // 2 else target - 1
        
        target_line = core.std.Crop(clip, 0, 0, target, h - target - 1)
        donor_line = core.std.Crop(clip, 0, 0, donor, h - donor - 1)
        
        fix_line = correction(target_line, donor_line, limit, curve)
        
        if target == 0:
            clip = core.std.StackVertical([fix_line, core.std.Crop(clip, 0, 0, 1, 0)])
        elif target == h - 1:
            clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, 1), fix_line])
        else:
            clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, h - target), fix_line, core.std.Crop(clip, 0, 0, target + 1, 0)])
        
        return clip
    
    def correction(target_line: VideoNode, donor_line: VideoNode, limit: int, curve: int) -> VideoNode:
        
        match abs(curve):
            case 0:
                expr = 'y.PlaneStatsAverage x.PlaneStatsAverage - x +'
            case 1:
                expr = 'y.PlaneStatsAverage x.PlaneStatsAverage / x *'
            case 2:
                expr = 'x y.PlaneStatsAverage log x.PlaneStatsAverage log / pow'
            case 3:
                expr = 'y.PlaneStatsAverage 1 x.PlaneStatsAverage / pow x pow'
            case _:
                raise ValueError(f'{func_name}: Please use -3...3 curve value')
        
        if curve < 0:
            target_line = core.std.Invert(target_line)
            donor_line = core.std.Invert(donor_line)
            limit = -limit
        
        target_line = core.std.PlaneStats(target_line)
        donor_line = core.std.PlaneStats(donor_line)
        
        fix_line = core.akarin.Expr([target_line, donor_line], expr)
        fix_line = core.std.RemoveFrameProps(fix_line, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
        
        if limit > 0:
            fix_line = mt_clamp(fix_line, target_line, target_line, limit, 0)
        elif limit < 0:
            fix_line = mt_clamp(fix_line, target_line, target_line, 0, -limit)
        
        if curve < 0:
            fix_line = core.std.Invert(fix_line)
        
        return fix_line
    
    defaults = ['x', 0, None, 0, 1, 0]
    
    for i in args:
        match i:
            case str() if i in {'x', 'y'}:
                i = [i] + defaults[1:]
            case list() if i and i[0] in {'x', 'y'}:
                if len(i) < 6:
                    i += defaults[len(i):]
                elif len(i) > 6:
                    raise ValueError(f'{func_name}: *args length must be <= 6 or *args must be "str"')
            case _:
                raise ValueError(f'{func_name}: *args must be "list" or "x|y"')
        
        clips[i[5]] = eval(f'axis_{i[0]}(clips[i[5]], *i[1:5])')
    
    if space == YUV:
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    else:
        clip = clips[0]
    
    return clip

def MaskDetail(clip: VideoNode, dx: float | None = None, dy: float | None = None, rg: int = 3, cutoff: int = 70,
                gain: float = 0.75, blur_more: bool = False, kernel: str = 'bilinear', b: float = 0, c: float = 0.5,
                taps: int = 3, frac: bool = True, down: bool = False, **after_args: Any) -> VideoNode:
    '''
    MaskDetail by "Tada no Snob", ported from AviSynth version with minor additions.
    Has nothing to do with the port by MonoS.
    It is based on the rescale class from muvsfunc, therefore it supports fractional resolutions
    and automatic width calculation based on the original aspect ratio.
    "down = True" is added for backward compatibility and does not support fractional resolutions.
    Also, this option is incompatible with using odd resolutions when there is chroma subsampling in the source.
    '''
    
    func_name = 'MaskDetail'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    if space == YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor
    w = clip.width
    h = clip.height
    
    if dy is None:
        dy = h * 2 // 3
    
    match kernel:
        case 'bicubic':
            rescaler = rescale.Bicubic(b, c)
        case 'lanczos':
            rescaler = rescale.Lanczos(taps)
        case _:
            rescaler = eval(f'rescale.{kernel.capitalize()}()')
    
    if dx is None:
        resc = rescaler.rescale(clip, dy, h if frac else None)
    else:
        resc = rescaler.descale(clip, dx, dy, h if frac else None)
        resc = rescaler.upscale(resc, w, h)
    
    mask = RemoveGrain(core.std.MakeDiff(clip, resc).hist.Luma(), rg)
    mask = core.akarin.Expr(mask, f'x {cutoff * factor} < 0 x {gain} {full} x + {full} / * * ?')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 2
    
    if 'inf_n' not in after_args:
        after_args['inf_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if down:
        if dx is None:
            raise ValueError(f'{func_name}: if "down" is "True", then "dx" can\'t be "None"')
        
        if not isinstance(dx, int) or not isinstance(dy, int):
            raise ValueError(f'{func_name}: if "down" is "True", then "dx" and "dy" must be "int"')
        
        if space == YUV and (dx >> sub_w << sub_w != dx or dy >> sub_h << sub_h != dy):
            raise ValueError(f'{func_name}: "dx" or "dy" does not match the chroma subsampling of the output clip')
        
        mask = core.resize.Bilinear(mask, dx, dy)
    
    if blur_more:
        mask = RemoveGrain(mask, 12)
    
    if space == YUV:
        mask = core.resize.Point(mask, format=format_id)
    
    return mask

def degrain_n(clip: VideoNode, *args: dict[str, Any], tr: int = 1, full_range: bool = False) -> VideoNode:
    '''
    Just an alias for mv.Degrain
    The parameters of individual functions are set as dictionaries. Unloading takes place sequentially, separated by commas.
    If you do not set anything, the default settings of MVTools itself apply.
    Function dictionaries are set in order: Super, Analyze, Degrain, Recalculate.
    Recalculate is optional, but you can specify several of them (as many as you want).
    If you need to specify settings for only one function, the rest of the dictionaries are served empty.
    '''
    
    func_name = 'degrain_n'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if tr > 6 or tr < 1:
        raise ValueError(f'{func_name}: 1 <= "tr" <= 6')
    
    if len(args) < 3:
        args += ({},) * (3 - len(args))
    
    if full_range:
        sup1 = Dither_Luma_Rebuild(clip, s0 = 1).mv.Super(rfilter=4, **args[0])
        sup2 = core.mv.Super(clip, levels=1, **args[0])
    else:
        sup1 = core.mv.Super(clip, **args[0])
    
    vectors = [core.mv.Analyse(sup1, isb=j, delta=i, **args[1]) for i in range(1, tr + 1) for j in (True, False)]
    
    for i in args[3:]:
        for j in range(tr * 2):
            vectors[j] = core.mv.Recalculate(sup1, vectors[j], **i)
    
    clip = eval(f'core.mv.Degrain{tr}(clip, sup2 if full_range else sup1, *vectors, **args[2])')
    
    return clip

def Destripe(clip: VideoNode, dx: int | None = None, dy: int | None = None, tff: bool = True, **descale_args: Any) -> VideoNode:
    '''
    Simplified Destripe from YomikoR without any unnecessary conversions and soapy EdgeFixer
    The internal Descale functions are unloaded as usual.
    The function values that differ for the upper and lower fields are indicated in the list.
    '''
    
    func_name = 'Destripe'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if dx is None:
        dx = clip.width
    
    if dy is None:
        dy = clip.height // 2
    
    second_args = {}
    
    for i in descale_args:
        if isinstance(descale_args[i], list):
            if len(descale_args[i]) == 2:
                second_args[i] = descale_args[i][1]
                descale_args[i] = descale_args[i][0]
            else:
                raise ValueError(f'{func_name}: {i} length must be 2')
        else:
            second_args[i] = descale_args[i]
    
    clip = core.std.SetFieldBased(clip, 0).std.SeparateFields(tff).std.SetFieldBased(0)
    fields = [clip[::2], clip[1::2]]
    
    fields[0] = core.descale.Descale(fields[0], dx, dy, **descale_args)
    fields[1] = core.descale.Descale(fields[1], dx, dy, **second_args)
    
    clip = core.std.Interleave(fields)
    clip = core.std.DoubleWeave(clip, tff)[::2]
    clip = core.std.SetFieldBased(clip, 0)
    
    return clip

def daa(clip: VideoNode, weight: float = 0.5, planes: int | list[int] | None = None, **znedi3_args: Any) -> VideoNode:
    '''
    daa by Didée, ported from AviSynth version with minor additions.
    '''
    
    func_name = 'daa'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {YUV, GRAY}:
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

def average_fields(clip: VideoNode, curve: int | list[int | None] = 1, weight: float = 0.5, mode: int = 0) -> VideoNode:
    '''
    Just an experiment. It leads to a common denominator of the average normalized values of the fields of one frame.
    Ideally, it should fix interlaced fades painlessly, but in practice this does not always happen.
    Apparently it depends on the source.
    '''
    
    func_name = 'average_fields'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    def simple_average(clip: VideoNode, curve: int | None, weight: float, mode: int) -> VideoNode:
        
        if curve is None:
            return clip
        
        if weight == 0:
            expr0 = 'x.PlaneStatsAverage'
        elif weight == 1:
            expr0 = 'y.PlaneStatsAverage'
        elif 0 < weight < 1:
            expr0 = f'x.PlaneStatsAverage {1 - weight} * y.PlaneStatsAverage {weight} * +'
        else:
            raise ValueError(f'{func_name}: 0 <= "weight" <= 1')
        
        match abs(curve):
            case 0:
                expr1 = f'{expr0} x.PlaneStatsAverage - x +'
                expr2 = f'{expr0} y.PlaneStatsAverage - y +'
            case 1:
                expr1 = f'{expr0} x.PlaneStatsAverage / x *'
                expr2 = f'{expr0} y.PlaneStatsAverage / y *'
            case 2:
                expr1 = f'x {expr0} log x.PlaneStatsAverage log / pow'
                expr2 = f'y {expr0} log y.PlaneStatsAverage log / pow'
            case 3:
                expr1 = f'{expr0} 1 x.PlaneStatsAverage / pow x pow'
                expr2 = f'{expr0} 1 y.PlaneStatsAverage / pow y pow'
            case _:
                raise ValueError(f'{func_name}: Please use -3...3 or "None" (only in the list) curve values')
        
        if curve < 0:
            clip = core.std.Invert(clip)
        
        match mode:
            case 0:
                clip = core.std.SeparateFields(clip, True).std.PlaneStats()
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
                clips = [core.std.Crop(clip, 0, 0, i, h - i - 1).std.PlaneStats() for i in range(h)]
                
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
        
        clip = core.std.RemoveFrameProps(clip, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
        
        return clip
    
    match curve:
        case int():
            curve = [curve] * num_p
        case list() if curve:
            if len(curve) < num_p:
                curve += [curve[-1]] * (num_p - len(curve))
            elif len(curve) > num_p:
                raise ValueError(f'{func_name}: "curve" must be shorter or the same length to number of planes, or "curve" must be "int"')
        case _:
            raise ValueError(f'{func_name}: "curve" must be "int" or list[int | None]')
    
    if space == YUV:
        clips = core.std.SplitPlanes(clip)
        
        for i in range(num_p):
            clips[i] = simple_average(clips[i], curve[i], weight, mode)
        
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    elif space == GRAY:
        clip = simple_average(clip, curve[0], weight, mode)
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    return clip

def nnedi3aas(clip: VideoNode, rg: int = 20, rep: int = 13, clamp: int = 0, planes: int | list[int] | None = None,
              **nnedi3_args: Any) -> VideoNode:
    '''
    nnedi2aas by Didée, ported from AviSynth version with minor additions.
    '''
    
    func_name = 'nnedi3aas'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {YUV, GRAY}:
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
        shrpD = core.std.MakeDiff(dbl, mt_clamp(dbl, RemoveGrain(dbl, [rg if i in planes else 0 for i in range(num_p)]),
                                  dbl, 0, clamp, planes=planes), planes=planes)
    else:
        shrpD = core.std.MakeDiff(dbl, RemoveGrain(dbl, [rg if i in planes else 0 for i in range(num_p)]), planes=planes)
    
    DD = Repair(shrpD, dblD, [rep if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes=planes)
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else dblD for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def dehalo_mask(clip: VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> VideoNode:
    '''
    Fork of jvsfunc.dehalo_mask from dnjulek with minor additions.
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.
    
    Args:
        src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
        expand: Expansion of edge mask.
        iterations: Protects parallel lines and corners that are usually damaged by YAHR.
        brz: Adjusts the internal line thickness.
        shift: Corrective shift for fine-tuning iterations
    '''
    
    func_name = 'dehalo_mask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if brz < 0 or brz > 255:
        raise ValueError(f'{func_name}: brz must be between 0 and 255')
    
    space = clip.format.color_family
    
    if space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    
    clip = core.akarin.Expr([clip, core.std.Maximum(clip).std.Maximum()], f'y x - {shift * factor} - 128 *')
    mask = core.tcanny.TCanny(clip, sigma=sqrt(expand * 2), mode=-1).akarin.Expr('x 16 *')
    
    for _ in range(iterations):
        clip = core.std.Maximum(clip)
    
    for _ in range(iterations):
        clip = core.std.Minimum(clip)
    
    clip = core.std.InvertMask(clip).std.BinarizeMask(80 * factor)
    
    if brz < 255:
        clip = core.std.Inflate(clip).std.Inflate().std.BinarizeMask(brz * factor)
    
    mask = core.akarin.Expr([mask, RemoveGrain(clip, 12, edges=True)], 'x y min')
    
    if space == YUV:
        mask = core.resize.Point(mask, format=format_id)
    
    return mask

def tp7_deband_mask(clip: VideoNode, thr: float | list[float] = 8, scale: float = 1, rg: bool = True, mt_prewitt: bool = False,
                    **after_args: Any) -> VideoNode:
    
    func_name = 'tp7_deband_mask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    if mt_prewitt:
        clip = Convolution(clip, 'prewitt', total=1 / scale)
    else:
        clip = core.std.Prewitt(clip, scale=scale)
    
    clip = mt_binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3, edges=True), 4, edges=True)
    
    if space == YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        w = clip.width
        h = clip.height
        
        clips = core.std.SplitPlanes(clip)
        clip = core.akarin.Expr(clips[1:], 'x y max')
        
        if sub_w > 0 or sub_h > 0:
            bits = clip.format.bits_per_sample
            
            clip = core.fmtc.resample(clip, w, h, kernel='spline', taps=6)
            if bits != 16:
                clip = core.fmtc.bitdepth(clip, bits=bits, dmode=1)
        
        clip = core.akarin.Expr([clip, clips[0]], 'x y max')
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space == YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def DeHalo_alpha(clip: VideoNode, rx: float = 2.0, ry: float = 2.0, darkstr: float = 1.0, brightstr: float = 1.0,
                 lowsens: float = 50, highsens: float = 50, ss: float = 1.5, showmask: bool = False) -> VideoNode:
    
    func_name = 'DeHalo_alpha'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor
    m4 = lambda var: max(int(var / 4 + 0.5) * 4, 16)
    
    halos = core.resize.Bicubic(clip, m4(w / rx), m4(h / ry)).resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    are = Convolution(clip, 'min-max')
    ugly = Convolution(halos, 'min-max')
    so = core.akarin.Expr([ugly, are], f'y x - y 0.001 + / {full - 1} * {lowsens * factor} - y {full} + {full * 2} / {highsens / 100} + *')
    lets = core.std.MaskedMerge(halos, clip, so)
    
    if ss == 1.0:
        remove = Repair(clip, lets, 1)
    else:
        remove = core.resize.Lanczos(clip, x := m4(w * ss), y := m4(h * ss), filter_param_a=3)
        remove = core.akarin.Expr([remove, core.std.Maximum(lets).resize.Bicubic(x, y)], 'x y min')
        remove = core.akarin.Expr([remove, core.std.Minimum(lets).resize.Bicubic(x, y)], 'x y max')
        remove = core.resize.Lanczos(remove, w, h, filter_param_a=3)
    
    clip = core.akarin.Expr([clip, remove], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        clip = core.resize.Point(so, format=orig.format.id) if space == YUV else so
    
    return clip

def FineDehalo(clip: VideoNode, rx: float = 2, ry: float | None = None, thmi: int = 80, thma: int = 128, thlimi: int = 50,
                thlima: int = 100, darkstr: float = 1.0, brightstr: float = 1.0, lowsens: float = 50, highsens: float = 50,
                ss: float = 1.25, showmask: int = 0, contra: float = 0.0, excl: bool = True, edgeproc: float = 0.0, mt_prewitt = False) -> VideoNode:
    
    func_name = 'FineDehalo'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
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
        xd = core.std.MakeDiff(bb, bb2).akarin.Expr(f'x {half} - 2.49 * {contra} * {half} +')
        xdd = core.akarin.Expr([xd, core.std.MakeDiff(clip, dehaloed)], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?')
        dehaloed = core.std.MergeDiff(dehaloed, xdd)
    
    if mt_prewitt:
        edges = Convolution(clip, 'prewitt')
    else:
        edges = core.std.Prewitt(clip)
    
    strong = core.akarin.Expr(edges, f'x {thmi} - {thma - thmi} / {full} *')
    large = mt_expand_multi(strong, sw=rx_i, sh=ry_i)
    light = core.akarin.Expr(edges, f'x {thlimi} - {thlima - thlimi} / {full} *')
    shrink = mt_expand_multi(light, mode='ellipse', sw=rx_i, sh=ry_i).akarin.Expr('x 4 *')
    shrink = mt_inpand_multi(shrink, mode='ellipse', sw=rx_i, sh=ry_i)
    shrink = RemoveGrain(RemoveGrain(shrink, 20), 20)
    outside = core.akarin.Expr([large, core.akarin.Expr([strong, shrink], 'x y max') if excl else strong], 'x y - 2 *')
    
    if edgeproc > 0:
        outside = core.akarin.Expr([outside, strong], f'x y {edgeproc * 0.66} * +')
    
    outside = core.akarin.Expr(RemoveGrain(outside, 20), 'x 2 *')
    
    clip = core.std.MaskedMerge(clip, dehaloed, outside)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        if showmask == 1:
            clip = core.resize.Point(outside, format=orig.format.id) if space == YUV else outside
        elif showmask == 2:
            clip = core.resize.Point(shrink, format=orig.format.id) if space == YUV else shrink
        elif showmask == 3:
            clip = core.resize.Point(edges, format=orig.format.id) if space == YUV else edges
        elif showmask == 4:
            clip = core.resize.Point(strong, format=orig.format.id) if space == YUV else strong
        else:
            raise ValueError(f'{func_name}: Please use 0...4 showmask value')
    
    return clip

def FineDehalo2(clip: VideoNode, hconv: list[int] | None = None, vconv: list[int] | None = None, showmask: bool = False) -> VideoNode:
    
    func_name = 'FineDehalo2'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if hconv is None:
        hconv = [-1, -2, 0, 0, 40, 0, 0, -2, -1]
    
    if vconv is None:
        vconv = [-2, -1, 0, 0, 40, 0, 0, -1, -2]
    
    def grow_mask(clip: VideoNode, mode: str) -> VideoNode:
        
        match mode:
            case 'v':
                coord = [0, 1, 0, 0, 0, 0, 1, 0]
            case 'h':
                coord = [0, 0, 0, 1, 1, 0, 0, 0]
        
        clip = core.std.Maximum(clip, coordinates=coord).std.Minimum(coordinates=coord)
        mask_1 = core.std.Maximum(clip, coordinates=coord)
        mask_2 = core.std.Maximum(mask_1, coordinates=coord).std.Maximum(coordinates=coord)
        clip = core.akarin.Expr([mask_2, mask_1], 'x y -')
        clip = core.akarin.Expr(RemoveGrain(clip, 12), 'x 1.8 *')
        
        return clip
    
    fix_h = Convolution(clip, [[1], vconv])
    fix_v = Convolution(clip, [hconv, [1]])
    mask_h = Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], saturate=0, total=4.0)
    mask_v = Convolution(clip, [1, 0, -1, 2, 0, -2, 1, 0, -1], saturate=0, total=4.0)
    temp_h = core.akarin.Expr([mask_h, mask_v], 'x 3 * y -')
    temp_v = core.akarin.Expr([mask_v, mask_h], 'x 3 * y -')
    
    mask_h = grow_mask(temp_h, 'v')
    mask_v = grow_mask(temp_v, 'h')
    
    clip = core.std.MaskedMerge(clip, fix_h, mask_h)
    clip = core.std.MaskedMerge(clip, fix_v, mask_v)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        clip = core.akarin.Expr([mask_h, mask_v], 'x y max')
        if space == YUV:
            clip = core.resize.Point(clip, format=orig.format.id)
    
    return clip

def InsaneAA(clip: VideoNode, ext_aa: VideoNode = None, ext_mask: VideoNode = None, desc_str: float = 0.3, mode: int = 1,
              kernel: str = 'bilinear', b: float = 1/3, c: float = 1/3, taps: int = 3, dx: int = None, dy: int = 720,
              dehalo: bool = False, masked: bool = False, frac: bool = True, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'InsaneAA'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
        orig_gray = clip
    elif space == GRAY:
        orig_gray = clip
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if ext_aa is None:
        w = clip.width
        h = clip.height
        
        match kernel:
            case 'bicubic':
                rescaler = rescale.Bicubic(b, c)
            case 'lanczos':
                rescaler = rescale.Lanczos(taps)
            case _:
                rescaler = eval(f'rescale.{kernel.capitalize()}()')
        
        if dx is None:
            dx = w / h * dy
        
        clip = rescaler.descale(clip, dx, dy, h if frac else None)
        clip_sp = core.resize.Spline36(clip, **rescaler.descale_args)
        clip = core.std.Merge(clip_sp, clip, desc_str)
        
        if dehalo:
            clip = FineDehalo(clip, thmi=45, thlimi=60, thlima=120, mt_prewitt=True)
        
        clip = rescaler.upscale(clip, w, h, partial(upscaler, **upscaler_args))
    elif isinstance(ext_aa, VideoNode) and ext_aa.format.color_family == GRAY:
        clip = ext_aa
    else:
        raise ValueError(f'{func_name}: The external AA should be GRAY')
    
    if masked:
        if ext_mask is None:
            ext_mask = core.std.Sobel(orig_gray, scale=2).std.Maximum()
        elif isinstance(ext_mask, VideoNode) and ext_mask.format.color_family == GRAY:
            pass
        else:
            raise ValueError(f'{func_name}: The external mask should be GRAY')
        
        clip = core.std.MaskedMerge(orig_gray, clip, ext_mask)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    return clip

def upscaler(clip: VideoNode, dx: int | None = None, dy: int | None = None, src_left: float | None = None, src_top: float | None = None,
             src_width: float | None = None, src_height: float | None = None, mode: int = 0, order: int = 0, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'upscaler'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w * 2
    
    if dy is None:
        dy = h * 2
    
    if src_left is None:
        src_left = 0
    
    if src_top is None:
        src_top = 0
    
    if src_width is None:
        src_width = w
    elif src_width <= 0:
        src_width += w - src_left
    
    if src_height is None:
        src_height = h
    elif src_height <= 0:
        src_height += h - src_top
    
    if dx > w * 2 or dy > h * 2:
        raise ValueError(f'{func_name}: upscale size is too big')
    
    def edi3_aa(clip: VideoNode, mode: int, order: bool, **upscaler_args: Any) -> VideoNode:
        
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
                eedi3_args = {i:upscaler_args[i] for i in signature(core.eedi3m.EEDI3).parameters if i in upscaler_args}
                znedi3_args = {i:upscaler_args[i] for i in signature(core.znedi3.nnedi3).parameters if i in upscaler_args}
                
                if any((x := i) not in eedi3_args and x not in znedi3_args for i in upscaler_args):
                    raise KeyError(f'{func_name}: Unsupported key {x} in upscaler_args')
                
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True, sclip=core.znedi3.nnedi3(clip, field=1, dh=True, **znedi3_args), **eedi3_args)
                clip = core.std.Transpose(clip)
                clip = core.eedi3m.EEDI3(clip, field=1, dh=True, sclip=core.znedi3.nnedi3(clip, field=1, dh=True, **znedi3_args), **eedi3_args)
            case _:
                raise ValueError(f'{func_name}: Please use 0...3 mode value')
        
        if not order:
            clip = core.std.Transpose(clip)
        
        return clip
    
    if mode:
        match order:
            case 0:
                clip = edi3_aa(clip, mode, True, **upscaler_args)
            case 1:
                clip = edi3_aa(clip, mode, False, **upscaler_args)
            case 2:
                clip = core.akarin.Expr([edi3_aa(clip, mode, True, **upscaler_args), edi3_aa(clip, mode, False, **upscaler_args)], 'x y max')
            case _:
                raise ValueError(f'{func_name}: Please use 0...2 order value')
        
        clip = autotap3(clip, dx, dy, src_left=src_left * 2 - 0.5, src_top=src_top * 2 - 0.5, src_width=src_width * 2, src_height=src_height * 2)
    else:
        kernel = upscaler_args.pop('kernel', 'spline36').capitalize()
        clip = eval(f'core.resize.{kernel}(clip, dx, dy, src_left=src_left, src_top=src_top, src_width=src_width, src_height=src_height, **upscaler_args)')
    
    return clip

def custom_mask(clip: VideoNode, mask: int = 0, scale: float = 1.0, boost: bool = False, offset: float = 0.0,
                **after_args: Any) -> VideoNode:
    
    func_name = 'custom_mask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    match mask:
        case 0:
            pass
        case 1:
            pass
        case 2:
            clip = core.akarin.Expr([Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], saturate=0, total=4.0),
                                     Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], saturate=0, total=4.0)],
                                     f'x y max {scale} *')
        case 3:
            clip = core.akarin.Expr([Convolution(clip, [8, 16, 8, 0, 0, 0, -8, -16, -8], saturate=0, total=8.0),
                                     Convolution(clip, [8, 0, -8, 16, 0, -16, 8, 0, -8], saturate=0, total=8.0)],
                                     f'x y max {scale} *')
        case _:
            raise ValueError(f'{func_name}: Please use 0...3 mask value')
    
    if boost:
        factor = 1 << clip.format.bits_per_sample - 8
        clip = core.akarin.Expr(clip, f'x {128 * factor} / 0.86 {offset} + pow {256 * factor - 1} *')
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    return clip

def diff_mask(first: VideoNode, second: VideoNode, thr: float = 8, scale: float = 1.0, rg: bool = True,
              mt_prewitt: bool | None = None, **after_args: Any) -> VideoNode:
    
    func_name = 'diff_mask'
    
    if any(not isinstance(i, VideoNode) for i in (first, second)):
        raise TypeError(f'{func_name} both clips must be of the VideoNode type')
    
    if first.format.sample_type != INTEGER or second.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if first.num_frames != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.bits_per_sample != second.format.bits_per_sample:
        raise ValueError(f'{func_name}: Sample types of clips do not match')
    
    space_f = first.format.color_family
    space_s = second.format.color_family
    
    if space_f == YUV:
        format_id = first.format.id
        first = core.std.ShufflePlanes(first, 0, GRAY)
    elif space_f == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the first clip')
    
    if space_s == YUV:
        second = core.std.ShufflePlanes(second, 0, GRAY)
    elif space_s == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the second clip')
    
    clip = core.akarin.Expr([first, second], 'x y - abs')
    
    if mt_prewitt is None:
        clip = core.akarin.Expr(clip, f'x {scale} *')
    else:
        if mt_prewitt:
            clip = Convolution(clip, 'prewitt', total=1 / scale)
        else:
            clip = core.std.Prewitt(clip, scale=scale)
    
    if thr:
        clip = mt_binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3, edges=True), 4, edges=True)
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def apply_range(first: VideoNode, second: VideoNode, *args: int | list[int]) -> VideoNode:
    
    func_name = 'apply_range'
    
    if any(not isinstance(i, VideoNode) for i in (first, second)):
        raise TypeError(f'{func_name} both clips must be of the VideoNode type')
    
    num_f = first.num_frames
    
    if num_f != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.name != second.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    for i in args:
        match i:
            case (int(a), int(b)) if 0 <= a < b and b < num_f:
                if a == 0:
                    first = second[:b + 1] + first[b + 1:]
                elif b == num_f - 1:
                    first = first[:a] + second[a:]
                else:
                    first = first[:a] + second[a:b + 1] + first[b + 1:]
            case int(a) | (int(a),) if 0 <= a < num_f:
                if a == 0:
                    first = second[a] + first[a + 1:]
                elif a == num_f - 1:
                    first = first[:a] + second[a]
                else:
                    first = first[:a] + second[a] + first[a + 1:]
            case _:
                raise ValueError(f'{func_name}: *args must be list[first_frame, last_frame], list[frame] or "int"')
    
    return first

def titles_mask(clip: VideoNode, thr: float = 230, rg: bool = True, **after_args: Any) -> VideoNode:
    
    func_name = 'titles_mask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    clip = mt_binarize(clip, thr)
    
    if rg:
        clip = RemoveGrain(RemoveGrain(clip, 3, edges=True), 4, edges=True)
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format=format_id)
    
    return clip

def after_mask(clip: VideoNode, flatten: int = 0, borders: list[int] | None = None, planes: int | list[int] | None = None,
               **after_args: int) -> VideoNode:
    
    func_name = 'after_mask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
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
    
    if flatten > 0:
        expr = ['x y max z max' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, flatten + 1):
            clip = core.akarin.Expr([clip, shift_clip(clip, -i), shift_clip(clip, i)], expr)
    elif flatten < 0:
        expr = ['x y min z min' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, -flatten + 1):
            clip = core.akarin.Expr([clip, shift_clip(clip, -i), shift_clip(clip, i)], expr)
    
    after_dict = dict(exp_n='Maximum', inp_n='Minimum', def_n='Deflate', inf_n='Inflate')
    
    for i in after_args:
        if i in after_dict:
            for _ in range(after_args[i]):
                clip = eval(f'core.std.{after_dict[i]}(clip, planes=planes)')
        else:
            raise KeyError(f'{func_name}: Unsupported key {i} in after_args')
    
    if borders:
        if len(borders) < 4:
            defaults = [0, clip.width - 1, 0, clip.height - 1]
            borders += defaults[len(borders):]
        elif len(borders) > 4:
            raise ValueError(f'{func_name}: borders length must be <= 4')
        
        factor = 1 << clip.format.bits_per_sample - 8
        
        expr = f'X {borders[0]} >= X {borders[1]} <= Y {borders[2]} >= Y {borders[3]} <= and and and {256 * factor - 1} 0 ? x min'
        clip = core.akarin.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def search_field_diffs(clip: VideoNode, mode: int | list[int] = 0, thr: float | list[float] = 0.001, div: float | list[float] = 2.0,
                       norm: bool = False, frames: list[int] | None = None, output: str | None = None, plane: int = 0) -> VideoNode:
    '''
    Search for deinterlacing failures after ftm/vfm and similar filters, the result is saved to a text file.
    
    The principle of operation is quite simple - each frame is divided into fields and absolute normalized difference is calculated
    for them using two different algorithms.
    
    Args:
        mode: function operation mode.
            0 and 1 - search for frames with absolute normalized difference above the specified threshold.
            2 and 3 - search for the absolute normalized difference change above the specified threshold.
            4 and 5 - search for single anomalies of absolute normalized difference changes above the specified threshold
            (n/p frame is skipped). Of the two possible values, the larger is compared with the threshold.
            The minimum ratio between the anomaly value and the change in adjacent, non-abnormal frames is specified by the div parameter.
            6 and 7 - search for double anomalies of absolute normalized difference changes above the specified threshold
            (both n/p frames are skipped). Of the four possible values, the largest is compared with the threshold.
            The minimum ratio between the anomaly value and the change in adjacent, non-abnormal frames is specified by the div parameter.
            In this case, the spread of the values of two abnormal frames must be strictly greater than the abnormal value.
            8 and 9 - debug mode for mode 4 and 5.
            10 and 11 - debug mode for mode 6 and 7.
            
            Even modes - the average normalized value is calculated for each field, and then their absolute difference.
            It is well suited for searching combo frames and interlaced fades.
            Odd modes - a classic algorithm, fields are subtracted from each other pixel by modulus,
            and the average normalized value is calculated for the resulting clip.
            It is well suited for detecting temporal anomalies.
            
            You can specify several modes as a list, in which case the result will be sorted by frame number, and within one frame by mode.
            Normal and debug modes cannot be mixed in one list. The default is "0".
        
        thr: the threshold for triggering the mode, it does not work for debug modes.
            You can specify several as a list, they will positionally correspond to the modes.
            If the thr list is less than the list of modes, the last thr value will work for all remaining modes. The default is "0.001".
        
        div: sets the minimum ratio between the anomaly value and the change in neighboring, non-abnormal frames.
            It is relevant for modes 4...7. You can specify several as a list, they will positionally correspond to the modes.
            If the div list is less than the list of modes, the last div value will work for all remaining modes. The default is "2.0".
        
        norm: normalization of absolute normalized difference values between 0 and 1. The default is "False".
        
        frames: a list of frames to check. The default is "all frames".
        
        output: path and name of the output file.
            By default, the file is created in the same directory where the application used for the analysis pass is located,
            the file name is "field_diffs.txt".
        
        plane: the position of the planar for calculating the absolute normalized difference. The default is "0" (luminance planar).
    '''
    
    func_name = 'search_field_diffs'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if plane not in set(range(clip.format.num_planes)):
        raise ValueError(f'{func_name}: Unsupported plane')
    
    match mode:
        case int() if mode in set(range(12)):
            mode = [mode]
        case list() if mode and (set(mode) <= set(range(8)) or set(mode) <= set(range(8, 12))):
            pass
        case _:
            raise ValueError(f'{func_name}: Please use 0...7 or 8...11 mode value or list[mode]')
    
    if output is None:
        output = f'field_diffs_mode({'_'.join(f'{i}' for i in mode)})_thr({'_'.join(f'{i}' for i in thr) if isinstance(thr, list) else thr}).txt'
    
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
    
    diffs = [[0] * num_f, [0] * num_f]
    
    def dump_diffs(n: int, f: list[VideoFrame], clip: VideoNode) -> VideoNode:
        
        nonlocal diffs
        
        diffs[0][n] = abs(f[0].props['PlaneStatsAverage'] - f[1].props['PlaneStatsAverage'])
        diffs[1][n] = f[0].props['PlaneStatsDiff']
        
        if n == num_f - 1:
            res = []
            
            dig = max(len(str(num_f)), 5)
            tab = max(len(str(i)) for i in thr)
            
            if norm:
                min_diffs_0 = min(diffs[0])
                max_diffs_0 = max(diffs[0])
                min_diffs_1 = min(diffs[1])
                max_diffs_1 = max(diffs[1])
                
                diffs[0] = [(i - min_diffs_0) / (max_diffs_0 - min_diffs_0) for i in diffs[0]]
                diffs[1] = [(i - min_diffs_1) / (max_diffs_1 - min_diffs_1) for i in diffs[1]]
            
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
                        file.write(f'{'frame':<{dig}} mode {'diff':<22} {'thr':<22} {'div':<8} thr2\n')
                    else:
                        file.write(f'{'frame':<{dig}} mode {'diff':<22} {'thr':<{tab}} div\n')
                    
                    file.writelines(res) if len(mode) == 1 else file.writelines(sorted(res))
            else:
                raise ValueError(f'{func_name}: there is no result, check the settings')
        
        return clip
    
    temp = core.std.SetFieldBased(clip, 0).std.SeparateFields(True)
    fields = [temp[::2], temp[1::2]]
    
    fields[0] = core.std.PlaneStats(*fields, plane=plane)
    fields[1] = core.std.PlaneStats(fields[1], plane=plane)
    
    clip = core.std.FrameEval(clip, partial(dump_diffs, clip=clip), prop_src=fields)
    
    return clip

def CombMask2(clip: VideoNode, cthresh: int | None = None, mthresh: int = 9, expand: bool = True, metric: int = 0,
              planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'CombMask2'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
        motionmask = core.akarin.Expr([clip, shift_clip(clip, 1)], [expr if i in planes else defaults[i] for i in range(num_p)])
        
        expr = 'x[0,1] x[0,-1] x max max y min'
        mask = core.akarin.Expr([motionmask, mask], [expr if i in planes else '' for i in range(num_p)])
    
    if expand:
        mask = core.std.Maximum(mask, planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    
    return mask

def mt_CombMask(clip: VideoNode, thr1: float = 10, thr2: float = 10, div: float = 256, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'mt_CombMask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if thr1 < 0 or thr2 < 0 or thr1 > 65535 or thr2 > 65535:
        raise ValueError(f'{func_name}: Please use 0...65535 thr1 and thr2 value')
    
    if thr1 > thr2:
        raise ValueError(f'{func_name}: thr1 must not be greater than thr2')
    
    if div <= 0:
        raise ValueError(f'{func_name}: div must be greater than zero')
    
    if clip.format.color_family not in {YUV, GRAY}:
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

def mt_binarize(clip: VideoNode, thr: float | list[float] = 128, upper: bool = False, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'mt_binarize'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    
    clip = core.akarin.Expr(clip, [expr[i] if i in planes else f'{128 * factor}' for i in range(num_p)])
    
    return clip

def delcomb(clip: VideoNode, thr1: float = 100, thr2: float = 5, mode: int = 0, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'delcomb'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    
    mask = mt_CombMask(clip, 7, 7, planes=0).std.Deflate(planes=0).std.Deflate(planes=0)
    mask = core.std.Minimum(mask, coordinates=[0, 0, 0, 1, 1, 0, 0, 0], planes=0)
    mask = mt_binarize(core.std.Maximum(mask, planes=0), thr1, planes=0).std.Maximum(planes=0)
    
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
    
    clip = core.akarin.Select([clip, filt], core.std.PlaneStats(mask), f'x.PlaneStatsAverage {thr2 * factor / (256 * factor - 1)} > 1 0 ?')
    
    return clip

def vinverse(clip: VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'vinverse'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    
    Vblur = Convolution(clip, [[1], [50, 99, 50]], planes=planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes=planes)
    
    expr0 = f'x x y - {sstr} * +'
    Vshrp = Convolution(Vblur, [[1], [1, 4, 6, 4, 1]], planes=planes)
    Vshrp = core.akarin.Expr([Vblur, Vshrp], [expr0 if i in planes else '' for i in range(num_p)])
    VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes=planes)
    
    expr1 = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD = core.akarin.Expr([VshrpD, VblurD], [expr1 if i in planes else '' for i in range(num_p)])
    
    res = core.std.MergeDiff(Vblur, VlimD, planes=planes)
    
    if amnt > 254:
        clip = res
    elif amnt == 0:
        pass
    else:
        amnt *= factor
        expr2 = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
        clip = core.akarin.Expr([clip, res], [expr2 if i in planes else '' for i in range(num_p)])
    
    return clip

def vinverse2(clip: VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'vinverse2'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    
    Vblur = sbrV(clip, planes=planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes=planes)
    
    expr0 = f'x x y - {sstr} * +'
    Vshrp = Convolution(Vblur, [[1], [1, 2, 1]], planes=planes)
    Vshrp  = core.akarin.Expr([Vblur, Vshrp], [expr0 if i in planes else '' for i in range(num_p)])
    VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes=planes)
    
    expr1 = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD  = core.akarin.Expr([VshrpD, VblurD], [expr1 if i in planes else '' for i in range(num_p)])
    
    res = core.std.MergeDiff(Vblur, VlimD, planes=planes)
    
    if amnt > 254:
        clip = res
    elif amnt == 0:
        pass
    else:
        amnt *= factor
        expr2 = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
        clip = core.akarin.Expr([clip, res], [expr2 if i in planes else '' for i in range(num_p)])
    
    return clip

def sbr(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'sbr'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
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
    rg11D = core.std.MakeDiff(clip, rg11, planes=planes)
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    rg11DD = Convolution(rg11D, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
    rg11DD = core.std.MakeDiff(rg11D, rg11DD, planes=planes)
    rg11DD = core.akarin.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, rg11DD, planes=planes)
    
    return clip

def sbrV(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'sbrV'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
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
    rg11D = core.std.MakeDiff(clip, rg11, planes=planes)
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    rg11DD = Convolution(rg11D, [[1], [1, 2, 1]], planes=planes)
    rg11DD = core.std.MakeDiff(rg11D, rg11DD, planes=planes)
    rg11DD = core.akarin.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, rg11DD, planes=planes)
    
    return clip

def avs_Blur(clip: VideoNode, amountH: float = 0, amountV: float | None = None, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'avs_Blur'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
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
    
    if amountV is None:
        amountV = amountH
    
    if amountH < -1 or amountV < -1 or amountH > 1.58 or amountV > 1.58:
        raise ValueError(f'{func_name}: the "amount" allowable range is from -1.0 to +1.58 ')
    
    center_h = 1 / 2 ** amountH
    side_h = (1 - 1 / 2 ** amountH) / 2
    
    center_v = 1 / 2 ** amountV
    side_v = (1 - 1 / 2 ** amountV) / 2
    
    expr = (f'x[-1,-1] x[-1,1] x[1,-1] x[1,1] + + + {side_h * side_v} * x[-1,0] x[1,0] + {side_h * center_v} * + '
            f'x[0,-1] x[0,1] + {center_h * side_v} * + x {center_h * center_v} * +')
    
    clip = core.akarin.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def avs_Sharpen(clip: VideoNode, amountH: float = 0, amountV: float | None = None, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'avs_Sharpen'
    
    if amountV is None:
        amountV = amountH
    
    if amountH < -1.58 or amountV < -1.58 or amountH > 1 or amountV > 1:
        raise ValueError(f'{func_name}: the "amount" allowable range is from -1.58 to +1.0 ')
    
    clip = avs_Blur(clip, -amountH, -amountV, planes)
    
    return clip

def mt_clamp(clip: VideoNode, bright_limit: VideoNode, dark_limit: VideoNode, overshoot: float = 0, undershoot: float = 0,
             planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'mt_clamp'
    
    if any(not isinstance(clip, VideoNode) for i in (clip, bright_limit, dark_limit)):
        raise TypeError(f'{func_name} all clips must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.id != bright_limit.format.id or clip.format.id != dark_limit.format.id:
        raise ValueError(f'{func_name}: "clip", "bright_limit" and "dark_limit" must have the same format')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    
    overshoot *= factor
    undershoot *= factor
    
    expr = f'x z {undershoot} - y {overshoot} + clamp'
    clip = core.akarin.Expr([clip, bright_limit, dark_limit], [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def MinBlur(clip: VideoNode, r: int = 1, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'MinBlur'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
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
            RG11D = core.std.MakeDiff(clip, RG11D, planes=planes)
            
            RG4D = RemoveGrain(clip, [4 if i in planes else 0 for i in range(num_p)])
            RG4D = core.std.MakeDiff(clip, RG4D, planes=planes)
        case 2:
            RG11D = RemoveGrain(clip, [11 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = core.std.MakeDiff(clip, RG11D, planes=planes)
            
            RG4D = core.ctmf.CTMF(clip, 2, planes=planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes=planes)
        case 3:
            RG11D = RemoveGrain(clip, [11 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = RemoveGrain(RG11D, [20 if i in planes else 0 for i in range(num_p)])
            RG11D = core.std.MakeDiff(clip, RG11D, planes=planes)
            
            RG4D = core.ctmf.CTMF(clip, 3, planes=planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes=planes)
        case _:
            raise ValueError(f'{func_name}: Please use 1...3 "r" value')
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    DD = core.akarin.Expr([RG11D, RG4D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, DD, planes=planes)
    
    return clip

def Dither_Luma_Rebuild(clip: VideoNode, s0: float = 2.0, c: float = 0.0625, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'Dither_Luma_Rebuild'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
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
    t = f'x {16 * factor} - {219 * factor} / 0 1 clamp'
    y = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + {256 * factor} *'
    uv = f'x {half} - 128 * 112 / {half} +'
    
    expr = [y] + [uv] * (num_p - 1)
    clip = core.akarin.Expr(clip, [expr[i] if i in planes else '' for i in range(num_p)])
    
    return clip

def mt_expand_multi(clip: VideoNode, mode: str = 'rectangle', sw: int = 1, sh: int = 1, planes: int | list[int] | None = None,
                    **thr_arg: float) -> VideoNode:
    
    func_name = 'mt_expand_multi'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
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
        if mode == 'losange' or (mode == 'ellipse' and sw % 3 != 1):
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0]
        else:
            mode_m = [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None
    
    if mode_m:
        clip = core.std.Maximum(clip, planes=planes, coordinates=mode_m, **thr_arg)
        clip = mt_expand_multi(clip, mode=mode, sw=sw - 1, sh=sh - 1, planes=planes, **thr_arg)
    
    return clip

def mt_inpand_multi(clip: VideoNode, mode: str = 'rectangle', sw: int = 1, sh: int = 1, planes: int | list[int] | None = None,
                    **thr_arg: float) -> VideoNode:
    
    func_name = 'mt_inpand_multi'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
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
        if mode == 'losange' or (mode == 'ellipse' and sw % 3 != 1):
            mode_m = [0, 1, 0, 1, 1, 0, 1, 0]
        else:
            mode_m = [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None
    
    if mode_m:
        clip = core.std.Minimum(clip, planes=planes, coordinates=mode_m, **thr_arg)
        clip = mt_inpand_multi(clip, mode=mode, sw=sw - 1, sh=sh - 1, planes=planes, **thr_arg)
    
    return clip

def avs_TemporalSoften(clip: VideoNode, radius: int = 0, scenechange: int = 0, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'avs_TemporalSoften'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    if radius < 0 or radius > 7:
        raise ValueError(f'{func_name}: Please use 0...7 "radius" value')
    
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
        clip = core.std.PlaneStats(clip, shift_clip(clip, -1))
        clip = core.akarin.PropExpr(clip, lambda: dict(_SceneChangeNext=f'x.PlaneStatsDiff {scenechange * factor / (256 * factor - 1)} > 1 0 ?'))
        clip = core.akarin.PropExpr([clip, shift_clip(clip, 1)], lambda: dict(_SceneChangePrev='y._SceneChangeNext'))
    
    if radius:
        clip = core.std.AverageFrames(clip, weights=[1] * (radius * 2 + 1), scenechange=bool(scenechange), planes=planes)
    
    if scenechange:
        clip = core.std.RemoveFrameProps(clip, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage',
                                                'PlaneStatsDiff', '_SceneChangeNext', '_SceneChangePrev'])
    
    return clip

def UnsharpMask(clip: VideoNode, strength: int = 64, radius: int = 3, threshold: int = 8, blur: str = 'box', roundoff: int = 0) -> VideoNode:
    '''
    Implementation of UnsharpMask with the ability to select the blur type (box or gauss) and rounding mode.
    By default, it perfectly imitates UnsharpMask from the WarpSharp package to AviSynth.
    '''
    
    func_name = 'UnsharpMask'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(strength, int) or strength < 0:
        raise TypeError(f'{func_name}: invalid "strength"')
    
    if not isinstance(radius, int) or radius < 0:
        raise TypeError(f'{func_name}: invalid "radius"')
    
    if not isinstance(threshold, int) or threshold < 0:
        raise TypeError(f'{func_name}: invalid "threshold"')
    
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
    
    num_p = clip.format.num_planes
    threshold <<= clip.format.bits_per_sample - 8
    side = radius * 2 + 1
    square = side ** 2
    
    match blur:
        case 'box':
            expr = (f'{' '.join(f'x[{j - radius},{i - radius}]' for i in range(side) for j in range(side))} '
                    f'{'+ ' * (square - 1)}{square} /{rnd} blur! x blur@ - abs {threshold} > x blur@ - {strength} * 128 /{rnd} x + x ?')
        case 'gauss':
            row = [x := (x * (side - i) // i if i != 0 else 1) for i in range(side)]
            matrix = [i * j for i in row for j in row]
            expr = (f'{' '.join(f'x[{j - radius},{i - radius}] {matrix[i * side + j]} *' for i in range(side) for j in range(side))} '
                    f'{'+ ' * (square - 1)}{sum(matrix)} /{rnd} blur! x blur@ - abs {threshold} > x blur@ - {strength} * 128 /{rnd} x + x ?')
        case _:
            raise ValueError(f'{func_name}: invalid "blur"')
    
    clip = core.akarin.Expr(clip, [expr] + [''] * (num_p - 1))
    
    return clip

def diff_tfm(clip: VideoNode, nc_clip: VideoNode, ovr_d: str, ovr_c: str, diff_proc: Callable[..., VideoNode] | None = None,
             planes: int | list[int] | None = None, **tfm_args) -> VideoNode:
    
    func_name = 'diff_tfm'
    
    if any(not isinstance(i, VideoNode) for i in (clip, nc_clip)):
        raise TypeError(f'{func_name} both clips must be of the VideoNode type')
    
    if clip.format.name != nc_clip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    num_f = clip.num_frames
    
    if num_f != nc_clip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if any(not isinstance(i, str) for i in (ovr_d, ovr_c)):
        raise TypeError(f'{func_name} both ovr\'s must be of the string type')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {YUV, GRAY}:
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
    
    diff = [core.akarin.Expr([clip_c, nc_clip_c], ['x y -' if i in planes else '' for i in range(num_p)]),
            core.akarin.Expr([clip_c, nc_clip_c], ['y x -' if i in planes else '' for i in range(num_p)])]
    
    match diff_proc:
        case None:
            pass
        case Callable():
            diff = [diff_proc(i, planes=planes) for i in diff]
        case _:
            raise TypeError(f'{func_name} invalid "diff_proc"')
    
    clip = core.akarin.Expr([nc_clip_d] + diff, ['x y z - +' if i in planes else '' for i in range(num_p)])
    
    if set(planes) != set(range(num_p)):
        clip = core.std.ShufflePlanes([clip if i in planes else clip_d for i in range(num_p)], list(range(num_p)), space)
    
    return clip

def diff_transfer(clip: VideoNode, nc_clip: VideoNode, target: VideoNode, diff_proc: Callable[..., VideoNode] | None = None,
                  planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'diff_transfer'
    
    if any(not isinstance(i, VideoNode) for i in (clip, nc_clip, target)):
        raise TypeError(f'{func_name} all clips must be of the VideoNode type')
    
    if clip.format.name != nc_clip.format.name or clip.format.name != target.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != nc_clip.num_frames or clip.num_frames != target.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if space not in {YUV, GRAY}:
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
    
    diff = [core.akarin.Expr([clip, nc_clip], ['x y -' if i in planes else '' for i in range(num_p)]),
            core.akarin.Expr([clip, nc_clip], ['y x -' if i in planes else '' for i in range(num_p)])]
    
    match diff_proc:
        case None:
            pass
        case Callable():
            diff = [diff_proc(i, planes=planes) for i in diff]
        case _:
            raise TypeError(f'{func_name} invalid "diff_proc"')
    
    clip = core.akarin.Expr([target] + diff, ['x y z - +' if i in planes else '' for i in range(num_p)])
    
    return clip

def shift_clip(clip: VideoNode, shift: int = 0, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'shift_clip'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
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
    
    with open(ovr_d, 'r') as file:
        for line in file:
            if (res := re.search(r'(\d+),(\d+) (\w+)', line)) is not None:
                first = int(res.group(1))
                last = int(res.group(2))
                seq = res.group(3)
                
                for i in range(first, last + 1):
                    frames_d[i] = seq[(i - first) % len(seq)]
                
            elif (res := re.search(r'(\d+) (\w)', line)) is not None:
                frames_d[int(res.group(1))] = res.group(2)
    
    with open(ovr_c, 'r') as file:
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
        if frames_d[i] != frames_c[i]:
            match frames_d[i]:
                case 'c':
                    match frames_c[i]:
                        case 'p':
                            result[0] += [i]
                        case 'u':
                            result[1] += [i]
                        case _:
                            raise ValueError(f'{func_name}: invalid "ovr_c" in frame {i}')
                case 'p':
                    match frames_c[i]:
                        case 'c':
                            result[0] += [i]
                        case 'u':
                            result[1] += [i]
                        case _:
                            raise ValueError(f'{func_name}: invalid "ovr_c" in frame {i}')
                case 'u':
                    match frames_c[i]:
                        case 'c':
                            result[1] += [i]
                        case 'p':
                            result[1] += [i]
                        case _:
                            raise ValueError(f'{func_name}: invalid "ovr_c" in frame {i}')
                case _:
                    raise ValueError(f'{func_name}: invalid "ovr_d" in frame {i}')
    
    return result

def RemoveGrain(clip: VideoNode, mode: int | list[int] = 2, edges: bool = False, roundoff: int = 1) -> VideoNode:
    '''
    Implementation of RgTools.RemoveGrain with clip edge processing and bank rounding.
    Supported modes: -1...28
    
    By default, the reference RemoveGrain is imitated, no edge processing is done (edges=False),
    arithmetic rounding is used (roundoff=1).
    '''
    
    func_name = 'RemoveGrain'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor - 1
    
    match mode:
        case int() if -1 <= mode <= 28:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and -1 <= i <= 28 for i in mode):
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
    
    expr = ['',
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
            'Y 1 bitand 0 = x[-1,-1] x[1,1] - abs dup d1! x[0,-1] x[0,1] - abs dup d2! x[1,-1] x[-1,1] - abs dup d3! min min dup '
            f'mind! d2@ = x[0,-1] x[0,1] + 2 /{rnd} mind@ d3@ = x[1,-1] x[-1,1] + 2 /{rnd} x[-1,-1] x[1,1] + 2 /{rnd} ? ? x ?',
            # mode 14
            'Y 1 bitand 1 = x[-1,-1] x[1,1] - abs dup d1! x[0,-1] x[0,1] - abs dup d2! x[1,-1] x[-1,1] - abs dup d3! min min dup '
            f'mind! d2@ = x[0,-1] x[0,1] + 2 /{rnd} mind@ d3@ = x[1,-1] x[-1,1] + 2 /{rnd} x[-1,-1] x[1,1] + 2 /{rnd} ? ? x ?',
            # mode 15
            'Y 1 bitand 0 = x[-1,-1] x[1,1] - abs dup d1! x[0,-1] x[0,1] - abs dup d2! x[1,-1] x[-1,1] - abs dup d3! min min dup '
            f'mind! x[-1,-1] x[0,-1] 2 * + x[1,-1] + x[-1,1] + x[0,1] 2 * + x[1,1] + 8 /{rnd} avg! d2@ = avg@ x[0,-1] x[0,1] min '
            'x[0,-1] x[0,1] max clamp mind@ d3@ = avg@ x[1,-1] x[-1,1] min x[1,-1] x[-1,1] max clamp avg@ x[-1,-1] x[1,1] min '
            'x[-1,-1] x[1,1] max clamp ? ? x ?',
            # mode 16
            'Y 1 bitand 1 = x[-1,-1] x[1,1] - abs dup d1! x[0,-1] x[0,1] - abs dup d2! x[1,-1] x[-1,1] - abs dup d3! min min dup '
            f'mind! x[-1,-1] x[0,-1] 2 * + x[1,-1] + x[-1,1] + x[0,1] 2 * + x[1,1] + 8 /{rnd} avg! d2@ = avg@ x[0,-1] x[0,1] min '
            'x[0,-1] x[0,1] max clamp mind@ d3@ = avg@ x[1,-1] x[-1,1] min x[1,-1] x[-1,1] max clamp avg@ x[-1,-1] x[1,1] min '
            'x[-1,-1] x[1,1] max clamp ? ? x ?',
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
            'x x[-1,-1] x[1,1] + 2 / trunc x[0,-1] x[0,1] + 2 / trunc min x[1,-1] x[-1,1] + 2 / trunc min x[-1,0] x[1,0] + 2 / '
            f'trunc min x[-1,-1] x[1,1] + 2 /{rnd} x[0,-1] x[0,1] + 2 /{rnd} max x[1,-1] x[-1,1] + 2 /{rnd} max x[-1,0] x[1,0] + '
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
            'mal2@ mil2@ - ld2! mal3@ mil3@ - ld3! mal4@ mil4@ - ld4! x x mal1@ - 0 max ld1@ dup1 - min x mal2@ - 0 max ld2@ '
            'dup1 - min max x mal3@ - 0 max ld3@ dup1 - min max x mal4@ - 0 max ld4@ dup1 - min max 0 max - 0 max mil1@ x - 0 '
            'max ld1@ dup1 - min mil2@ x - 0 max ld2@ dup1 - min max mil3@ x - 0 max ld3@ dup1 - min max mil4@ x - 0 max ld4@ '
            f'dup1 - min max 0 max + {full} min',
            # mode 25
            f'x x[-1,0] < {full} x x[-1,0] - ? x x[1,0] < {full} x x[1,0] - ? min x x[-1,-1] < {full} x x[-1,-1] - ? min x '
            f'x[0,-1] < {full} x x[0,-1] - ? min x x[1,-1] < {full} x x[1,-1] - ? min x x[-1,1] < {full} x x[-1,1] - ? min x '
            f'x[0,1] < {full} x x[0,1] - ? min x x[1,1] < {full} x x[1,1] - ? min mn! x[-1,0] x < {full} x[-1,0] x - ? x[1,0] x '
            f'< {full} x[1,0] x - ? min x[-1,-1] x < {full} x[-1,-1] x - ? min x[0,-1] x < {full} x[0,-1] x - ? min x[1,-1] x < '
            f'{full} x[1,-1] x - ? min x[-1,1] x < {full} x[-1,1] x - ? min x[0,1] x < {full} x[0,1] x - ? min x[1,1] x < {full} '
            f'x[1,1] x - ? min pl! x pl@ 2 / trunc mn@ pl@ - 0 max min + {full} min mn@ 2 / trunc pl@ mn@ - 0 max min - 0 max',
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
            'upper! x lower@ upper@ min lower@ upper@ max clamp',
            # mode -1
            f'{128 * factor}']
    
    orig = clip
    
    clip = core.akarin.Expr(clip, [expr[i] for i in mode])
    
    if not edges:
        clip = core.akarin.Expr([clip, orig], 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?')
    
    return clip

def Repair(clip: VideoNode, refclip: VideoNode, mode: int | list[int] = 2, edges: bool = False, roundoff: int = 1) -> VideoNode:
    '''
    Implementation of RgTools.Repair with clip edge processing and bank rounding.
    Supported modes: -1...28
    
    By default, the reference Repair is imitated, no edge processing is done (edges=False),
    arithmetic rounding is used (roundoff=1).
    '''
    
    func_name = 'Repair'
    
    if any(not isinstance(i, VideoNode) for i in (clip, refclip)):
        raise TypeError(f'{func_name} both clips must be of the VideoNode type')
    
    if clip.format.name != refclip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != refclip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor - 1
    
    match mode:
        case int() if -1 <= mode <= 28:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and -1 <= i <= 28 for i in mode):
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
    
    expr = ['',
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
            'd6! x y[0,1] - abs d7! x y[1,1] - abs d8! x y - abs dx! d1@ d2@ min d3@ min d4@ min d5@ min d6@ min d7@ min d8@ min '
            'dx@ min mind! mind@ d7@ = y[0,1] mind@ d8@ = y[1,1] mind@ d6@ = y[-1,1] mind@ d2@ = y[0,-1] mind@ d3@ = y[1,-1] '
            'mind@ d1@ = y[-1,-1] mind@ d5@ = y[1,0] mind@ dx@ = y y[-1,0] ? ? ? ? ? ? ? ?',
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
            'd6! y y[0,1] - abs d7! y y[1,1] - abs d8! d1@ d2@ max d1@ d2@ min dup mind! d3@ clamp mind@ d3@ min dup mind! d4@ '
            'clamp mind@ d4@ min dup mind! d5@ clamp mind@ d5@ min dup mind! d6@ clamp mind@ d6@ min dup mind! d7@ clamp mind@ '
            f'd7@ min d8@ clamp maxd! x y maxd@ - 0 max y maxd@ + {full} min clamp',
            # mode 21
            'y[-1,-1] y[1,1] max y - 0 max y y[-1,-1] y[1,1] min - 0 max max y[0,-1] y[0,1] max y - 0 max y y[0,-1] y[0,1] min - '
            '0 max max min y[1,-1] y[-1,1] max y - 0 max y y[1,-1] y[-1,1] min - 0 max max min y[-1,0] y[1,0] max y - 0 max y '
            f'y[-1,0] y[1,0] min - 0 max max min minu! x y minu@ - 0 max y minu@ + {full} min clamp',
            # mode 22
            'x y[-1,-1] - abs x y[0,-1] - abs min x y[1,-1] - abs min x y[-1,0] - abs min x y[1,0] - abs min x y[-1,1] - abs min '
            f'x y[0,1] - abs min x y[1,1] - abs min mind! y x mind@ - 0 max x mind@ + {full} min clamp',
            # mode 23
            'x y[-1,-1] - abs d1! x y[0,-1] - abs d2! x y[1,-1] - abs d3! x y[-1,0] - abs d4! x y[1,0] - abs d5! x y[-1,1] - abs '
            'd6! x y[0,1] - abs d7! x y[1,1] - abs d8! d1@ d2@ max d1@ d2@ min dup mind! d3@ clamp mind@ d3@ min dup mind! d4@ '
            'clamp mind@ d4@ min dup mind! d5@ clamp mind@ d5@ min dup mind! d6@ clamp mind@ d6@ min dup mind! d7@ clamp mind@ '
            f'd7@ min d8@ clamp maxd! y x maxd@ - 0 max x maxd@ + {full} min clamp',
            # mode 24
            'y[-1,-1] y[1,1] max x - 0 max x y[-1,-1] y[1,1] min - 0 max max y[0,-1] y[0,1] max x - 0 max x y[0,-1] y[0,1] min - '
            '0 max max min y[1,-1] y[-1,1] max x - 0 max x y[1,-1] y[-1,1] min - 0 max max min y[-1,0] y[1,0] max x - 0 max x '
            f'y[-1,0] y[1,0] min - 0 max max min minu! y x minu@ - 0 max x minu@ + {full} min clamp',
            # mode 25
            f'{128 * factor}',
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
            'upper! x lower@ upper@ min y min lower@ upper@ max y max clamp',
            # mode -1
            f'{128 * factor}']
    
    orig = clip
    
    clip = core.akarin.Expr([clip, refclip], [expr[i] for i in mode])
    
    if not edges:
        clip = core.akarin.Expr([clip, orig], 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?')
    
    return clip

def TemporalRepair(clip: VideoNode, refclip: VideoNode, mode: int = 0, edges: bool = False, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'TemporalRepair'
    
    if any(not isinstance(i, VideoNode) for i in (clip, refclip)):
        raise TypeError(f'{func_name} both clips must be of the VideoNode type')
    
    if clip.format.name != refclip.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    if clip.num_frames != refclip.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if not isinstance(mode, int) or mode < 0 or mode > 4:
        raise ValueError(f'{func_name}: invalid "mode"')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = ['x a z min y min a z max y max clamp',
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
            'clamp ?']
    
    orig = clip
    
    clip = clip[0] + core.akarin.Expr([clip, refclip, shift_clip(clip, 1), shift_clip(clip, -1)], [expr[mode] if i in planes else '' for i in range(num_p)])[1:-1] + clip[-1]
    
    if not edges and mode in {1, 2, 3}:
        clip = core.akarin.Expr([clip, orig], 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?')
    
    return clip

def Clense(clip: VideoNode, previous: VideoNode | None = None, next: VideoNode | None = None, reduceflicker: bool = False,
           planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'Clense'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    if previous is None:
        previous = shift_clip(clip, 1)
    elif isinstance(previous, VideoNode) and clip.format.name == previous.format.name and clip.num_frames == previous.num_frames:
        pass
    else:
        raise TypeError(f'{func_name}: invalid "previous"')
    
    if next is None:
        next = shift_clip(clip, -1)
    elif isinstance(next, VideoNode) and clip.format.name == next.format.name and clip.num_frames == next.num_frames:
        pass
    else:
        raise TypeError(f'{func_name}: invalid "next"')
    
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
    
    clip = clip[0] + core.akarin.Expr([clip, previous, next], [expr if i in planes else '' for i in range(num_p)])[1:-1] + clip[-1]
    
    if reduceflicker:
        clip = clip[0:2] + core.akarin.Expr([orig, shift_clip(clip, 1), next], [expr if i in planes else '' for i in range(num_p)])[2:-1] + clip[-1]
    
    return clip

def BackwardClense(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'BackwardClense'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = f'x y z min 2 * z - 0 max y z max 2 * z - {full} min clamp'
    
    clip = clip[:2] + core.akarin.Expr([clip, shift_clip(clip, 1), shift_clip(clip, 2)], [expr if i in planes else '' for i in range(num_p)])[2:]
    
    return clip

def ForwardClense(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'ForwardClense'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1
    
    match planes:
        case None:
            planes = list(range(num_p))
        case int() if planes in set(range(num_p)):
            planes = [planes]
        case list() if 0 < len(planes) <= num_p and set(planes) <= set(range(num_p)):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "planes"')
    
    expr = f'x y z min 2 * z - 0 max y z max 2 * z - {full} min clamp'
    
    clip = core.akarin.Expr([clip, shift_clip(clip, -1), shift_clip(clip, -2)], [expr if i in planes else '' for i in range(num_p)])[:-2] + clip[-2:]
    
    return clip

def VerticalCleaner(clip: VideoNode, mode: int | list[int] = 1, edges: bool = False) -> VideoNode:
    
    func_name = 'VerticalCleaner'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1
    
    match mode:
        case int() if 0 <= mode <= 2:
            mode = [mode]
        case list() if 0 < len(mode) <= num_p and all(isinstance(i, int) and 0 <= i <= 2 for i in mode):
            pass
        case _:
            raise ValueError(f'{func_name}: invalid "mode"')
    
    if not isinstance(edges, bool):
        raise TypeError(f'{func_name}: invalid "edges"')
    
    expr = ['',
            # mode 1
            'x[0,-1] x[0,1] min x max x[0,-1] x[0,1] max min',
            # mode 2
            'x x[0,-1] x[0,1] min x[0,-1] x[0,-2] x[0,-1] - 0 max - 0 max x[0,1] x[0,2] x[0,1] - 0 max - 0 max max min x[0,-1] '
            f'x[0,-2] - 0 max x[0,-1] + {full} min x[0,1] x[0,2] - 0 max x[0,1] + {full} min min x[0,-1] max x[0,1] max clamp']
    
    orig = clip
    
    clip = core.akarin.Expr(clip, [expr[i] for i in mode])
    
    if not edges:
        expr = ['',
                # mode 1
                'Y 0 = Y height 1 - = or y x ?',
                # mode 2
                'Y 1 <= Y height 2 - >= or y x ?']
        
        clip = core.akarin.Expr([clip, orig], [expr[i] for i in mode])
    
    return clip

def Convolution(clip: VideoNode, mode: str | list[int] | list[list[int]] | None = None, saturate: int | None = None,
                total: float | None = None, planes: int | list[int] | None = None) -> VideoNode:
    '''
    An unnatural hybrid of std.Convolution, mt_convolution and mt_edge.
    
    All named modes from mt_edge are present. The kernel can also be specified as two flat matrices or a square matrix.
    Unlike std.Convolution, it works correctly with edges.
    The default mode value is boxblur 3x3.
    The default value of saturate is 1.
    The default value of total is the sum of the absolute values of the resulting matrix.
    For named modes, the default values are changed to obtain the desired result, but they can be overridden by specifying them explicitly.
    '''
    
    func_name = 'Convolution'
    
    if not isinstance(clip, VideoNode):
        raise TypeError(f'{func_name} the clip must be of the VideoNode type')
    
    if clip.format.sample_type != INTEGER:
        raise TypeError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family not in {YUV, GRAY}:
        raise TypeError(f'{func_name}: Unsupported color family')
    
    num_p = clip.format.num_planes
    full = (1 << clip.format.bits_per_sample) - 1
    
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
        case ((int(), *a), (int(), *b)) if all(isinstance(i, int) for i in a + b) and len(mode[0]) % 2 == 1 and len(mode[1]) % 2 == 1:
            side_h, side_v = len(mode[0]), len(mode[1])
            mode = [j * i for i in mode[1] for j in mode[0]]
        case (int(), *a) if all(isinstance(i, int) for i in a) and (side_h := int(sqrt(len(mode)))) ** 2 == len(mode) and side_h % 2 == 1:
            side_v = side_h
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
        case 'min-max':
            expr = f'x y -{'' if total is None else f' {total} /'}'
            return core.akarin.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], [expr if i in planes else '' for i in range(num_p)])
        case 'hprewitt':
            return core.akarin.Expr([Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], 0 if saturate is None else saturate, 1.0 if total is None else total, planes),
                                     Convolution(clip, [1, 0, -1, 2, 0, -2, 1, 0, -1], 0 if saturate is None else saturate, 1.0 if total is None else total, planes)],
                                     ['x y max' if i in planes else '' for i in range(num_p)])
        case 'prewitt':
            return core.akarin.Expr([Convolution(clip, [1, 1, 0, 1, 0, -1, 0, -1, -1], 0 if saturate is None else saturate, 1.0 if total is None else total, planes),
                                     Convolution(clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], 0 if saturate is None else saturate, 1.0 if total is None else total, planes),
                                     Convolution(clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], 0 if saturate is None else saturate, 1.0 if total is None else total, planes),
                                     Convolution(clip, [0, -1, -1, 1, 0, -1, 1, 1, 0], 0 if saturate is None else saturate, 1.0 if total is None else total, planes)],
                                     ['x y max z a max max' if i in planes else '' for i in range(num_p)])
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
    
    expr = (f'{' '.join(f'x[{j - (side_h // 2)},{i - (side_v // 2)}] {mode[i * side_h + j]} *' for i in range(side_v) for j in range(side_h))} '
            f'{'+ ' * (len(mode) - 1)}{div} /{fix}')
    
    clip = core.akarin.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip
