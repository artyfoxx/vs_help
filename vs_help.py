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
    rg_fix
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
    MTCombMask
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
'''

from vapoursynth import core, GRAY, YUV, VideoNode, VideoFrame, INTEGER
from muvsfunc import rescale
from typing import Any
from math import sqrt
from functools import partial
from inspect import signature

def autotap3(clip: VideoNode, dx: int | None = None, dy: int | None = None, mtaps3: int = 1, thresh: int = 256, **crop_args: float) -> VideoNode:
    '''
    Lanczos-based resize from "*.mp4 guy", ported from AviSynth version with minor modifications.
    In comparison with the original, processing accuracy has been doubled, support for 8-16 bit depth
    and crop parameters has been added, and dead code has been removed.
    
    dx and dy are the desired resolution. The other parameters are not documented in any way and are selected using the poke method.
    Cropping options are added as **kwargs. The key names are the same as in VapourSynth-resize.
    '''
    
    func_name = 'autotap3'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
            raise ValueError(f'{func_name}: Unsupported key {x} in crop_args')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    t1 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 1, **crop_args)
    t2 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 2, **crop_args)
    t3 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 3, **crop_args)
    t4 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 4, **crop_args)
    t5 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 5, **crop_args)
    t6 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 9, **crop_args)
    t7 = core.resize.Lanczos(clip, dx, dy, filter_param_a = 36, **crop_args)
    
    m1 = core.std.Expr([clip, core.resize.Lanczos(t1, w, h, filter_param_a = 1, **back_args)], 'x y - abs')
    m2 = core.std.Expr([clip, core.resize.Lanczos(t2, w, h, filter_param_a = 1, **back_args)], 'x y - abs')
    m3 = core.std.Expr([clip, core.resize.Lanczos(t3, w, h, filter_param_a = 1, **back_args)], 'x y - abs')
    m4 = core.std.Expr([clip, core.resize.Lanczos(t4, w, h, filter_param_a = 2, **back_args)], 'x y - abs')
    m5 = core.std.Expr([clip, core.resize.Lanczos(t5, w, h, filter_param_a = 2, **back_args)], 'x y - abs')
    m6 = core.std.Expr([clip, core.resize.Lanczos(t6, w, h, filter_param_a = 3, **back_args)], 'x y - abs')
    m7 = core.std.Expr([clip, core.resize.Lanczos(t7, w, h, filter_param_a = 6, **back_args)], 'x y - abs')
    
    expr = f'x y - {thresh} *'
    
    cp1 = core.std.MaskedMerge(avs_Blur(t1, 1.42), t2, core.std.Expr([m1, m2], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m100 = core.std.Expr([clip, core.resize.Bilinear(cp1, w, h, **back_args)], 'x y - abs')
    cp2 = core.std.MaskedMerge(cp1, t3, core.std.Expr([m100, m3], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m101 = core.std.Expr([clip, core.resize.Bilinear(cp2, w, h, **back_args)], 'x y - abs')
    cp3 = core.std.MaskedMerge(cp2, t4, core.std.Expr([m101, m4], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m102 = core.std.Expr([clip, core.resize.Bilinear(cp3, w, h, **back_args)], 'x y - abs')
    cp4 = core.std.MaskedMerge(cp3, t5, core.std.Expr([m102, m5], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m103 = core.std.Expr([clip, core.resize.Bilinear(cp4, w, h, **back_args)], 'x y - abs')
    cp5 = core.std.MaskedMerge(cp4, t6, core.std.Expr([m103, m6], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m104 = core.std.Expr([clip, core.resize.Bilinear(cp5, w, h, **back_args)], 'x y - abs')
    clip = core.std.MaskedMerge(cp5, t7, core.std.Expr([m104, m7], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy, **crop_args)], [*range(orig.format.num_planes)], space)
    
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
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
        raise ValueError(f'{func_name}: Unsupported color family')
    
    fd1 = core.resize.Lanczos(clip, dx, dy, filter_param_a = mtaps1)
    fre1 = core.resize.Lanczos(fd1, w, h, filter_param_a = mtaps1)
    fre2 = autotap3(fre1, x := max(w // 16 * 8, 144), y := max(h // 16 * 8, 144), mtaps3, athresh)
    fre2 = autotap3(fre2, w, h, mtaps3, athresh)
    m1 = core.std.Expr([fre1, clip], f'x y - abs {thresh} - {thresh2} *')
    m2 = core.resize.Lanczos(core.resize.Lanczos(core.frfun7.Frfun7(m1, 2.01, 256, 256), x, y, filter_param_a = ttaps), dx, dy, filter_param_a = ttaps)
    
    d = core.std.MaskedMerge(clip, fre2, m1) if preblur else clip
    d2 = autotap3(d, dx, dy, mtaps3, athresh)
    d3 = core.resize.Lanczos(core.resize.Lanczos(d, w, h, filter_param_a = ttaps), dx, dy, filter_param_a = ttaps)
    d4 = core.std.MaskedMerge(core.std.Expr([d2, d3],  f'x y - {sharp1} * x +'), core.std.Expr([d2, d3],  f'y x - {blur1} * x +'), m2)
    d5 = autotap3(d4, w, h, mtaps3, athresh)
    
    e = autotap3(core.std.MaskedMerge(d5, clip, m1), dx, dy, mtaps3, athresh)
    e = core.warp.AWarpSharp2(e, thresh = wthresh, blur = wblur, depth = depth)
    e = core.warp.AWarpSharp2(e, thresh = wthresh, blur = wblur, depth = depth)
    e = core.warp.AWarpSharp2(e, thresh = wthresh, blur = wblur, depth = depth)
    e = core.warp.AWarpSharp2(e, thresh = wthresh, blur = wblur, depth = depth)
    
    fd12 = core.resize.Lanczos(e, dx ** 2 // w // 16 * 16, dy ** 2 // h // 16 * 16, filter_param_a = mtaps2)
    fre12 = core.resize.Lanczos(fd12, dx, dy, filter_param_a = mtaps2)
    m12 = core.std.Expr([fre12, e], f'x y - abs {thresh} - {thresh2} *')
    m12 = core.resize.Lanczos(m12, dx // 16 * 8, dy // 16 * 8, filter_param_a = mtaps2).resize.Lanczos(dx, dy, filter_param_a = mtaps2)
    
    e2 = core.resize.Lanczos(core.resize.Lanczos(e, w, h, filter_param_a = ltaps), dx, dy, filter_param_a = ltaps)
    e2 = core.warp.AWarpSharp2(e2, thresh = wthresh, blur = wblur, depth = depth)
    e2 = core.warp.AWarpSharp2(e2, thresh = wthresh, blur = wblur, depth = depth)
    e2 = core.warp.AWarpSharp2(e2, thresh = wthresh, blur = wblur, depth = depth)
    e2 = core.warp.AWarpSharp2(e2, thresh = wthresh, blur = wblur, depth = depth)
    
    e3 = core.std.MaskedMerge(core.std.Expr([e, e2], f'y x - {blur2} * x +'), core.std.Expr([e, e2], f'x y - {sharp2} * x +'), m12)
    e3 = core.warp.AWarpSharp2(e3, thresh = wthresh, blur = wblur, depth = depth)
    e3 = core.warp.AWarpSharp2(e3, thresh = wthresh, blur = wblur, depth = depth)
    e3 = core.warp.AWarpSharp2(e3, thresh = wthresh, blur = wblur, depth = depth)
    e3 = core.warp.AWarpSharp2(e3, thresh = wthresh, blur = wblur, depth = depth)
    
    clip = core.std.MaskedMerge(d4, e3, m2)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy)], [*range(orig.format.num_planes)], space)
    
    return clip

def bion_dehalo(clip: VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1, m: bool = False) -> VideoNode:
    '''
    Dehalo by bion, ported from AviSynth version with minor additions.
    mode = 1, 5, 11 - the weakest, artifacts will not cause.
    mode = 2, 3, 4 - bad modes, eat innocent parts, can't be used.
    mode = 10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
    mode = 14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.
    '''
    
    func_name = 'bion_dehalo'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    e1 = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], f'x y - {4 * factor} - 4 *')
    e2 = core.std.Maximum(e1).std.Maximum()
    e2 = core.std.Merge(e2, core.std.Maximum(e2)).std.Inflate()
    e3 = core.std.Expr([core.std.Merge(e2, core.std.Maximum(e2)), core.std.Deflate(e1)], 'x y 1.2 * -').std.Inflate()
    
    m0 = core.std.Expr([clip, core.std.BoxBlur(clip, hradius = 2, vradius = 2)], 'x y - abs 0 > x y - 0.3125 * x + x ?')
    m1 = core.std.Expr([clip, m0], f'x y - {factor} - 128 *').std.Maximum().std.Inflate()
    m2 = core.std.Maximum(m1).std.Maximum()
    m3 = core.std.Expr([m1, m2], 'y x -').rgvs.RemoveGrain(21).std.Maximum()
    
    match mask:
        case 1:
            pass
        case 2:
            e3 = m3
        case 3:
            e3 = core.std.Expr([e3, m3], 'x y min')
        case 4:
            e3 = core.std.Expr([e3, m3], 'x y max')
        case _:
            raise ValueError(f'{func_name}: Please use 1...4 mask value')
    
    blurr = MinBlur(clip, 1).std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if rg:
        dh1 = core.std.MaskedMerge(core.rgvs.Repair(clip, core.rgvs.RemoveGrain(clip, 21), 1), blurr, e3)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, e3)
    
    dh1D = core.std.MakeDiff(clip, dh1)
    tmp = sbr(dh1)
    med2D = core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2))
    DD  = core.std.Expr([dh1D, med2D], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs 2 * < x y {half} - 2 * {half} + ? ?')
    dh2 = core.std.MergeDiff(dh1, DD)
    
    clip = mt_clamp(clip, core.rgvs.Repair(clip, dh2, mode) if rep else dh2, clip, 0, 20)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if m:
        clip = core.resize.Point(e3, format = orig.format.id) if space == YUV else e3
    
    return clip

def fix_border(clip: VideoNode, *args: str | list[str | int | None]) -> VideoNode:
    '''
    A simple functions for fix brightness artifacts at the borders of the frame.
    All values are set as positional string arguments. The strings have the following format:
    [axis, target, donor, limit, curve, plane]. Only axis is mandatory.
    axis - can take the values "x" or "y" for columns and rows, respectively.
    target - the target column/row, it is counted from the upper left edge of the screen, by default 0.
    donor - the donor column/row, by default "None" (is calculated automatically as one closer to the center of the frame).
    limit - by default 0, without restrictions, positive values prohibit the darkening of target rows/columns
    and limit the maximum lightening, negative values - on the contrary, it's set in 8-bit notation.
    curve - target correction curve, by default 1, 0 - subtraction and addition, -1 and 1 - division and multiplication,
    -2 and 2 - logarithm and exponentiation, -3 and 3 - nth root and exponentiation.
    plane - by default 0.
    '''
    
    func_name = 'fix_border'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        num_p = clip.format.num_planes
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
    elif space == GRAY:
        clips = [clip]
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
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
            case str():
                i = [i] + defaults[1:]
            case list():
                if len(i) < 6:
                    i += defaults[len(i):]
                elif len(i) > 6:
                    raise ValueError(f'{func_name}: *args length must be <= 6 or *args must be "str"')
            case _:
                raise ValueError(f'{func_name}: *args must be "list" or "str"')
        
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
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
        raise ValueError(f'{func_name}: Unsupported color family')
    
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
    
    mask = core.std.MakeDiff(clip, resc).hist.Luma()
    mask = rg_fix(mask, rg)
    mask = core.std.Expr(mask, f'x {cutoff * factor} < 0 x {gain} {full} x + {full} / * * ?')
    
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
        mask = core.std.Convolution(mask, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if space == YUV:
        mask = core.resize.Point(mask, format = format_id)
    
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
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if tr > 6 or tr < 1:
        raise ValueError(f'{func_name}: 1 <= "tr" <= 6')
    
    if len(args) < 3:
        args += ({},) * (3 - len(args))
    
    if full_range:
        sup1 = Dither_Luma_Rebuild(clip, s0 = 1).mv.Super(rfilter = 4, **args[0])
        sup2 = core.mv.Super(clip, levels = 1, **args[0])
    else:
        sup1 = core.mv.Super(clip, **args[0])
    
    vectors = [core.mv.Analyse(sup1, isb = j, delta = i, **args[1]) for i in range(1, tr + 1) for j in (True, False)]
    
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
    
    clip = core.std.SetFieldBased(clip, 0)
    clip = core.std.SeparateFields(clip, tff)
    clip = core.std.SetFieldBased(clip, 0)
    fields = [clip[::2], clip[1::2]]
    
    fields[0] = core.descale.Descale(fields[0], dx, dy, **descale_args)
    fields[1] = core.descale.Descale(fields[1], dx, dy, **second_args)
    
    clip = core.std.Interleave(fields)
    clip = core.std.DoubleWeave(clip, tff)[::2]
    clip = core.std.SetFieldBased(clip, 0)
    
    return clip

def daa(clip: VideoNode, planes: int | list[int] | None = None, **znedi3_args: Any) -> VideoNode:
    '''
    daa by Didée, ported from AviSynth version with minor additions.
    '''
    
    func_name = 'daa'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [0.5 if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1] if clip.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]
    shrpD = core.std.MakeDiff(dbl, core.std.Convolution(dbl, matrix, planes = planes), planes = planes)
    DD = core.rgvs.Repair(shrpD, dblD, [13 if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip

def average_fields(clip: VideoNode, curve: int | list[int | None] = 1, weight: float = 0.5, mode: int = 0) -> VideoNode:
    '''
    Just an experiment. It leads to a common denominator of the average normalized values of the fields of one frame.
    Ideally, it should fix interlaced fades painlessly, but in practice this does not always happen.
    Apparently it depends on the source.
    '''
    
    func_name = 'average_fields'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
                            clips[i], clips[i + 1] = core.akarin.Expr([clips[i], clips[i + 1]], expr1), \
                                                     core.akarin.Expr([clips[i], clips[i + 1]], expr2)
                
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
        case list():
            if len(curve) < num_p:
                curve += [curve[-1]] * (num_p - len(curve))
            elif len(curve) > num_p:
                raise ValueError(f'{func_name}: "curve" must be shorter or the same length to number of planes, or "curve" must be "int"')
        case _:
            raise ValueError(f'{func_name}: "curve" must be "int" or list[int | None]')
    
    if space == YUV:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        
        for i in range(num_p):
            clips[i] = simple_average(clips[i], curve[i], weight, mode)
        
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    elif space == GRAY:
        clip = simple_average(clip, curve[0], weight, mode)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    return clip

def rg_fix(clip: VideoNode, mode: int | list[int] = 2) -> VideoNode:
    '''
    Alias for RemoveGrain. Redirects obsolete modes to internal functions-analogues of VapourSynth.
    '''
    
    func_name = 'rg_fix'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    def simple_fix(clip: VideoNode, mode: int) -> VideoNode:
        
        match mode:
            case 0:
                pass
            case 4:
                clip = core.std.Median(clip)
            case 11 | 12:
                clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])
            case 19:
                clip = core.std.Convolution(clip, [1, 1, 1, 1, 0, 1, 1, 1, 1])
            case 20:
                clip = core.std.Convolution(clip, [1, 1, 1, 1, 1, 1, 1, 1, 1])
            case _:
                clip = core.rgvs.RemoveGrain(clip, mode)
        
        return clip
    
    match mode:
        case int():
            mode = [mode] * num_p
        case list():
            if len(mode) < num_p:
                mode += [mode[-1]] * (num_p - len(mode))
            elif len(mode) > num_p:
                raise ValueError(f'{func_name}: "mode" must be shorter or the same length to number of planes, or "mode" must be "int"')
        case _:
            raise ValueError(f'{func_name}: "mode" must be list[int] or "int"')
    
    if space == YUV:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        
        for i in range(num_p):
            clips[i] = simple_fix(clips[i], mode[i])
        
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    elif space == GRAY:
        clip = simple_fix(clip, mode[0])
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    return clip

def znedi3aas(clip: VideoNode, rg: int = 20, rep: int = 13, clamp: int = 0, planes: int | list[int] | None = None,
              **znedi3_args: Any) -> VideoNode:
    '''
    nnedi2aas by Didée, ported from AviSynth version with minor additions.
    '''
    
    func_name = 'znedi3aas'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [0.5 if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    
    if clamp > 0:
        shrpD = core.std.MakeDiff(dbl, mt_clamp(dbl, rg_fix(dbl, [rg if i in planes else 0 for i in range(num_p)]),
                                  dbl, 0, clamp, planes = planes), planes = planes)
    else:
        shrpD = core.std.MakeDiff(dbl, rg_fix(dbl, [rg if i in planes else 0 for i in range(num_p)]), planes = planes)
    
    DD = core.rgvs.Repair(shrpD, dblD, [rep if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip

def dehalo_mask(clip: VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> VideoNode:
    '''
    Fork of jvsfunc.dehalo_mask from dnjulek with minor additions.
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.
    :param src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    :param expand: Expansion of edge mask.
    :param iterations: Protects parallel lines and corners that are usually damaged by YAHR.
    :param brz: Adjusts the internal line thickness.
    :param shift: Corrective shift for fine-tuning iterations
    '''
    
    func_name = 'dehalo_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if brz < 0 or brz > 255:
        raise ValueError(f'{func_name}: brz must be between 0 and 255')
    
    space = clip.format.color_family
    
    if space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    
    clip = core.std.Expr([clip, core.std.Maximum(clip).std.Maximum()], f'y x - {shift * factor} - 128 *')
    mask = core.tcanny.TCanny(clip, sigma = sqrt(expand * 2), mode = -1).std.Expr('x 16 *')
    
    for _ in range(iterations):
        clip = core.std.Maximum(clip)
    
    for _ in range(iterations):
        clip = core.std.Minimum(clip)
    
    clip = core.std.InvertMask(clip).std.BinarizeMask(80 * factor)
    
    if brz < 255:
        clip = core.std.Inflate(clip).std.Inflate().std.BinarizeMask(brz * factor)
    
    clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    mask = core.std.Expr([mask, clip], 'x y min')
    
    if space == YUV:
        mask = core.resize.Point(mask, format = format_id)
    
    return mask

def tp7_deband_mask(clip: VideoNode, thr: float | list[float] = 8, scale: float = 1, rg: bool = True, mt_prewitt: bool = False,
                    **after_args: Any) -> VideoNode:
    
    func_name = 'tp7_deband_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if mt_prewitt:
        clip = custom_mask(clip, 1, scale)
    else:
        clip = core.std.Prewitt(clip, scale = scale)
    
    clip = mt_binarize(clip, thr, planes = [*range(num_p)])
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if space == YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        w = clip.width
        h = clip.height
        
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        clip = core.std.Expr(clips[1:], 'x y max')
        
        if sub_w > 0 or sub_h > 0:
            bits = clip.format.bits_per_sample
            
            clip = core.fmtc.resample(clip, w, h, kernel = 'spline', taps = 6)
            if bits != 16:
                clip = core.fmtc.bitdepth(clip, bits = bits, dmode = 1)
        
        clip = core.std.Expr([clip, clips[0]], 'x y max')
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip

def DeHalo_alpha(clip: VideoNode, rx: float = 2.0, ry: float = 2.0, darkstr: float = 1.0, brightstr: float = 1.0,
                 lowsens: float = 50, highsens: float = 50, ss: float = 1.5, showmask: bool = False) -> VideoNode:
    
    func_name = 'DeHalo_alpha'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
        raise ValueError(f'{func_name}: Unsupported color family')
    
    factor = 1 << clip.format.bits_per_sample - 8
    full = 256 * factor
    m4 = lambda var: max(int(var / 4 + 0.5) * 4, 16)
    
    halos = core.resize.Bicubic(clip, m4(w / rx), m4(h / ry), filter_param_a = 1/3, filter_param_b = 1/3)
    halos = core.resize.Bicubic(halos, w, h, filter_param_a = 1, filter_param_b = 0)
    are = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], 'x y -')
    ugly = core.std.Expr([core.std.Maximum(halos), core.std.Minimum(halos)], 'x y -')
    so = core.std.Expr([ugly, are], f'y x - y {0.001 * factor} + / {full - 1} * {lowsens * factor} - y {full} + {full * 2} / {highsens / 100} + *')
    lets = core.std.MaskedMerge(halos, clip, so)
    
    if ss == 1.0:
        remove = core.rgvs.Repair(clip, lets, 1)
    else:
        remove = core.resize.Lanczos(clip, x := m4(w * ss), y := m4(h * ss), filter_param_a = 3)
        remove = core.std.Expr([remove, core.std.Maximum(lets).resize.Bicubic(x, y, filter_param_a = 1/3, filter_param_b = 1/3)], 'x y min')
        remove = core.std.Expr([remove, core.std.Minimum(lets).resize.Bicubic(x, y, filter_param_a = 1/3, filter_param_b = 1/3)], 'x y max')
        remove = core.resize.Lanczos(remove, w, h, filter_param_a = 3)
    
    clip = core.std.Expr([clip, remove], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if showmask:
        clip = core.resize.Point(so, format = orig.format.id) if space == YUV else so
    
    return clip

def FineDehalo(clip: VideoNode, rx: float = 2, ry: float | None = None, thmi: int = 80, thma: int = 128, thlimi: int = 50,
                thlima: int = 100, darkstr: float = 1.0, brightstr: float = 1.0, lowsens: float = 50, highsens: float = 50,
                ss: float = 1.25, showmask: int = 0, contra: float = 0.0, excl: bool = True, edgeproc: float = 0.0, mt_prewitt = False) -> VideoNode:
    
    func_name = 'FineDehalo'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
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
        
        bb = core.std.Convolution(dehaloed, [1, 2, 1, 2, 4, 2, 1, 2, 1])
        bb2 = core.rgvs.Repair(bb, core.rgvs.Repair(bb, core.ctmf.CTMF(bb, 2), 1), 1)
        xd = core.std.MakeDiff(bb, bb2).std.Expr(f'x {half} - 2.49 * {contra} * {half} +')
        xdd = core.std.Expr([xd, core.std.MakeDiff(clip, dehaloed)], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?')
        dehaloed = core.std.MergeDiff(dehaloed, xdd)
    
    if mt_prewitt:
        edges = custom_mask(clip, 1)
    else:
        edges = core.std.Prewitt(clip)
    
    strong = core.std.Expr(edges, f'x {thmi} - {thma - thmi} / {full} *')
    large = mt_expand_multi(strong, sw = rx_i, sh = ry_i)
    light = core.std.Expr(edges, f'x {thlimi} - {thlima - thlimi} / {full} *')
    shrink = mt_expand_multi(light, mode = 'ellipse', sw = rx_i, sh = ry_i).std.Expr('x 4 *')
    shrink = mt_inpand_multi(shrink, mode = 'ellipse', sw = rx_i, sh = ry_i)
    shrink = core.std.Convolution(shrink, [1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Convolution([1, 1, 1, 1, 1, 1, 1, 1, 1])
    outside = core.std.Expr([large, core.std.Expr([strong, shrink], 'x y max') if excl else strong], 'x y - 2 *')
    
    if edgeproc > 0:
        outside = core.std.Expr([outside, strong], f'x y {edgeproc * 0.66} * +')
    
    outside = core.std.Convolution(outside, [1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Expr('x 2 *')
    
    clip = core.std.MaskedMerge(clip, dehaloed, outside)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if showmask:
        if showmask == 1:
            clip = core.resize.Point(outside, format = orig.format.id) if space == YUV else outside
        elif showmask == 2:
            clip = core.resize.Point(shrink, format = orig.format.id) if space == YUV else shrink
        elif showmask == 3:
            clip = core.resize.Point(edges, format = orig.format.id) if space == YUV else edges
        elif showmask == 4:
            clip = core.resize.Point(strong, format = orig.format.id) if space == YUV else strong
        else:
            raise ValueError(f'{func_name}: Please use 0...4 showmask value')
    
    return clip

def FineDehalo2(clip: VideoNode, hconv: list[int] | None = None, vconv: list[int] | None = None, showmask: bool = False) -> VideoNode:
    
    func_name = 'FineDehalo2'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    if hconv is None:
        hconv = [-1, -2, 0, 0, 40, 0, 0, -2, -1]
    
    if vconv is None:
        vconv = [-2, -1, 0, 0, 40, 0, 0, -1, -2]
    
    def grow_mask(clip: VideoNode, mode: str) -> VideoNode:
        
        if mode == 'v':
            coord = [0, 1, 0, 0, 0, 0, 1, 0]
        else:
            coord = [0, 0, 0, 1, 1, 0, 0, 0]
        
        clip = core.std.Maximum(clip, coordinates = coord).std.Minimum(coordinates = coord)
        mask_1 = core.std.Maximum(clip, coordinates = coord)
        mask_2 = core.std.Maximum(mask_1, coordinates = coord).std.Maximum(coordinates = coord)
        clip = core.std.Expr([mask_2, mask_1], 'x y -')
        clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Expr('x 1.8 *')
        
        return clip
    
    fix_h = core.std.Convolution(clip, vconv, mode = 'v')
    fix_v = core.std.Convolution(clip, hconv, mode = 'h')
    mask_h = core.std.Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], divisor = 4, saturate = False)
    mask_v = core.std.Convolution(clip, [1, 0, -1, 2, 0, -2, 1, 0, -1], divisor = 4, saturate = False)
    temp_h = core.std.Expr([mask_h, mask_v], 'x 3 * y -')
    temp_v = core.std.Expr([mask_v, mask_h], 'x 3 * y -')
    
    mask_h = grow_mask(temp_h, 'v')
    mask_v = grow_mask(temp_v, 'h')
    
    clip = core.std.MaskedMerge(clip, fix_h, mask_h)
    clip = core.std.MaskedMerge(clip, fix_v, mask_v)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if showmask:
        clip = core.std.Expr([mask_h, mask_v], 'x y max')
        if space == YUV:
            clip = core.resize.Point(clip, format = orig.format.id)
    
    return clip

def InsaneAA(clip: VideoNode, ext_aa: VideoNode = None, ext_mask: VideoNode = None, desc_str: float = 0.3, mode: int = 1,
              kernel: str = 'bilinear', b: float = 1/3, c: float = 1/3, taps: int = 3, dx: int = None, dy: int = 720,
              dehalo: bool = False, masked: bool = False, frac: bool = True, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'InsaneAA'
    
    clip = core.std.SetFieldBased(clip, 0)
    
    space = clip.format.color_family
    
    if space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
        orig_gray = clip
    elif space == GRAY:
        orig_gray = clip
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
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
            clip = FineDehalo(clip, thmi = 45, thlimi = 60, thlima = 120, mt_prewitt = True)
        
        clip = rescaler.upscale(clip, w, h, partial(upscaler, **upscaler_args))
    elif ext_aa.format.color_family == GRAY:
        clip = ext_aa
    else:
        raise ValueError(f'{func_name}: The external AA should be GRAY')
    
    if masked:
        if ext_mask is None:
            ext_mask = core.std.Sobel(orig_gray, scale = 2).std.Maximum()
        elif ext_mask.format.color_family == GRAY:
            pass
        else:
            raise ValueError(f'{func_name}: The external mask should be GRAY')
        
        clip = core.std.MaskedMerge(orig_gray, clip, ext_mask)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    return clip

def upscaler(clip: VideoNode, dx: int | None = None, dy: int | None = None, src_left: float | None = None, src_top: float | None = None,
             src_width: float | None = None, src_height: float | None = None, mode: int = 0, order: int = 0, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'upscaler'
    
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
                clip = core.znedi3.nnedi3(clip, field = 1, dh = True, **upscaler_args)
                clip = core.std.Transpose(clip)
                clip = core.znedi3.nnedi3(clip, field = 1, dh = True, **upscaler_args)
            case 2:
                clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, **upscaler_args)
                clip = core.std.Transpose(clip)
                clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, **upscaler_args)
            case 3:
                eedi3_args = {i:upscaler_args[i] for i in signature(core.eedi3m.EEDI3).parameters if i in upscaler_args}
                znedi3_args = {i:upscaler_args[i] for i in signature(core.znedi3.nnedi3).parameters if i in upscaler_args}
                
                if any((x := i) not in eedi3_args and x not in znedi3_args for i in upscaler_args):
                    raise ValueError(f'{func_name}: Unsupported key {x} in upscaler_args')
                
                clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, sclip = core.znedi3.nnedi3(clip, field = 1, dh = True, **znedi3_args), **eedi3_args)
                clip = core.std.Transpose(clip)
                clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, sclip = core.znedi3.nnedi3(clip, field = 1, dh = True, **znedi3_args), **eedi3_args)
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
                clip = core.std.Expr([edi3_aa(clip, mode, True, **upscaler_args), edi3_aa(clip, mode, False, **upscaler_args)], 'x y max')
            case _:
                raise ValueError(f'{func_name}: Please use 0...2 order value')
        
        clip = autotap3(clip, dx, dy, src_left = src_left * 2 - 0.5, src_top = src_top * 2 - 0.5, src_width = src_width * 2, src_height = src_height * 2)
    else:
        kernel = upscaler_args.pop('kernel', 'spline36').capitalize()
        clip = eval(f'core.resize.{kernel}(clip, dx, dy, src_left = src_left, src_top = src_top, src_width = src_width, src_height = src_height, **upscaler_args)')
    
    return clip

def custom_mask(clip: VideoNode, mask: int = 0, scale: float = 1.0, boost: bool = False, offset: float = 0.0,
                **after_args: Any) -> VideoNode:
    
    func_name = 'custom_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    match mask:
        case 0:
            pass
        case 1:
            clip = core.std.Expr([core.std.Convolution(clip, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor = 1, saturate = False),
                                  core.std.Convolution(clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor = 1, saturate = False),
                                  core.std.Convolution(clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor = 1, saturate = False),
                                  core.std.Convolution(clip, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor = 1, saturate = False)],
                                  f'x y max z a max max {scale} *')
        case 2:
            clip = core.std.Expr([core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor = 4, saturate = False),
                                  core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor = 4, saturate = False)],
                                  f'x y max {scale} *')
        case 3:
            clip = core.std.Expr([core.std.Convolution(clip, [8, 16, 8, 0, 0, 0, -8, -16, -8], divisor = 8, saturate = False),
                                  core.std.Convolution(clip, [8, 0, -8, 16, 0, -16, 8, 0, -8], divisor = 8, saturate = False)],
                                  f'x y max {scale} *')
        case _:
            raise ValueError(f'{func_name}: Please use 0...3 mask value')
    
    if boost:
        factor = 1 << clip.format.bits_per_sample - 8
        clip = core.std.Expr(clip, f'x {128 * factor} / 0.86 {offset} + pow {(256 * factor) - 1} *')
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    return clip

def diff_mask(first: VideoNode, second: VideoNode, thr: float = 8, scale: float = 1.0, rg: bool = True,
              **after_args: Any) -> VideoNode:
    
    func_name = 'diff_mask'
    
    if first.format.sample_type != INTEGER or second.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
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
    
    clip = core.std.Expr([first, second], f'x y - abs {scale} *')
    
    if thr:
        clip = mt_binarize(clip, thr)
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip

def apply_range(first: VideoNode, second: VideoNode, *args: int | list[int]) -> VideoNode:
    
    func_name = 'apply_range'
    
    num_f = first.num_frames
    
    if num_f != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.name != second.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    for i in args:
        match i:
            case int():
                i = [i]
            case list():
                pass
            case _:
                raise ValueError(f'{func_name}: *args must be list[int] or int')
        
        if (x := i[0]) < 0 or (x := i[-1]) >= num_f:
            raise ValueError(f'{func_name}: {x} is out of frame range')
        
        match len(i):
            case 2:
                if i[0] >= i[1]:
                    raise ValueError(f'{func_name}: {i[0]} must not be equal to or greater than {i[1]}')
                
                if i[0] == 0:
                    first = second[:i[1] + 1] + first[i[1] + 1:]
                elif i[1] == num_f - 1:
                    first = first[:i[0]] + second[i[0]:]
                else:
                    first = first[:i[0]] + second[i[0]:i[1] + 1] + first[i[1] + 1:]
            case 1:
                if i[0] == 0:
                    first = second[i[0]] + first[i[0] + 1:]
                elif i[0] == num_f - 1:
                    first = first[:i[0]] + second[i[0]]
                else:
                    first = first[:i[0]] + second[i[0]] + first[i[0] + 1:]
            case _:
                raise ValueError(f'{func_name}: *args must be list[first_frame, last_frame], list[frame] or "int"')
    
    return first

def titles_mask(clip: VideoNode, thr: float = 230, rg: bool = True, **after_args: Any) -> VideoNode:
    
    func_name = 'titles_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    elif space == GRAY:
        pass
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    clip = mt_binarize(clip, thr)
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if after_args:
        clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip

def after_mask(clip: VideoNode, flatten: int = 0, borders: list[int] | None = None, planes: int | list[int] | None = None,
               **after_args: int) -> VideoNode:
    
    func_name = 'after_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    if flatten > 0:
        expr = ['x y max z max' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], expr)
    elif flatten < 0:
        expr = ['x y min z min' if i in planes else '' for i in range(num_p)]
        
        for i in range(1, -flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], expr)
    
    after_dict = {'exp_n': 'Maximum', 'inp_n': 'Minimum', 'def_n': 'Deflate', 'inf_n': 'Inflate'}
    
    for i in after_args:
        if i in after_dict:
            for _ in range(after_args[i]):
                clip = eval(f'core.std.{after_dict[i]}(clip, planes = planes)')
        else:
            raise ValueError(f'{func_name}: Unsupported key {i} in after_args')
    
    if borders:
        if len(borders) < 4:
            defaults = [0, clip.width - 1, 0, clip.height - 1]
            borders += defaults[len(borders):]
        elif len(borders) > 4:
            raise ValueError(f'{func_name}: borders length must be <= 4')
        
        factor = 1 << clip.format.bits_per_sample - 8
        
        expr = f'X {borders[0]} >= X {borders[1]} <= Y {borders[2]} >= Y {borders[3]} <= and and and {(256 * factor) - 1} 0 ? x min'
        clip = core.akarin.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def search_field_diffs(clip: VideoNode, thr: float = 0.001, div: float = 2, mode: int = 0, output: str | None = None,
                       plane: int = 0) -> VideoNode:
    
    func_name = 'search_field_diffs'
    
    if mode < 0 or mode > 7:
        raise ValueError(f'{func_name}: Please use 0...7 mode value')
    
    if output is None:
        output = f'field_diffs_mode_{mode}_thr_{thr:.0e}.txt'
    
    if div <= 0:
        raise ValueError(f'{func_name}: div must be greater than zero')
    
    num_f = clip.num_frames
    field_diffs = [0.0] * num_f
    
    def dump_diffs(n: int, f: list[VideoFrame], clip: VideoNode) -> VideoNode:
        
        field_diffs[n] = f[0].props['PlaneStatsDiff'] if mode & 1 else abs(f[0].props['PlaneStatsAverage'] - f[1].props['PlaneStatsAverage'])
        
        if n == num_f - 1:
            match mode // 2:
                case 0:
                    result = (f'{i} {x:.20f}\n' for i in range(num_f) if (x := field_diffs[i]) >= thr)
                case 1:
                    result = (f'{i} {x:.20f}\n' for i in range(1, num_f) if (x := abs(field_diffs[i - 1] - field_diffs[i])) >= thr)
                case 2:
                    result = (f'{i} {x:.20f}\n' for i in range(1, num_f - 1) if (x := max(abs(field_diffs[i - 1] - field_diffs[i]),
                              abs(field_diffs[i] - field_diffs[i + 1]))) >= thr and abs(field_diffs[i - 1] - field_diffs[i + 1]) <= x / div)
                case _:
                    result = (f'{i} {x:.20f}\n' for i in range(1, num_f - 2) if (x := max(abs(field_diffs[i - 1] - field_diffs[i]),
                              abs(field_diffs[i + 1] - field_diffs[i + 2]), abs(field_diffs[i - 1] - field_diffs[i + 1]),
                              abs(field_diffs[i] - field_diffs[i + 2]))) >= thr and abs(field_diffs[i - 1] - field_diffs[i + 2]) <= x / div
                              and abs(field_diffs[i] - field_diffs[i + 1]) > x)
            
            with open(output, 'w') as file:
                file.writelines(result)
        
        return clip
    
    temp = core.std.SeparateFields(clip, True)
    fields = [temp[::2], temp[1::2]]
    
    fields[0] = core.std.PlaneStats(*fields, plane = plane)
    fields[1] = core.std.PlaneStats(fields[1], plane = plane)
    
    clip = core.std.FrameEval(clip, partial(dump_diffs, clip = clip), prop_src = fields)
    
    return clip

def MTCombMask(clip: VideoNode, thr1: float = 30, thr2: float = 30, div: float = 256, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'MTCombMask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if thr1 < 0 or thr2 < 0 or thr1 > 65535 or thr2 > 65535:
        raise ValueError(f'{func_name}: Please use 0...65535 thr1 and thr2 value')
    
    if thr1 > thr2:
        raise ValueError(f'{func_name}: thr1 must not be greater than thr2')
    
    if div <= 0:
        raise ValueError(f'{func_name}: div must be greater than zero')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    power = factor ** 2
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    expr = f'x[0,-1] x - x[0,1] x - * var! var@ {thr1 * power} < 0 var@ {thr2 * power} > {256 * factor - 1} var@ {div * factor} / ? ?'
    clip = core.akarin.Expr(clip, [expr if i in planes else f'{128 * factor}' for i in range(num_p)])
    
    return clip

def mt_binarize(clip: VideoNode, thr: float | list[float] = 128, upper: bool = False, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'mt_binarize'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    match thr:
        case int() | float():
            thr = [thr] * num_p
        case list():
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

def delcomb(clip: VideoNode, thr1: float = 100, thr2: float = 5, mode: int = 0, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'delcomb'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    mask = MTCombMask(clip, 7, 7, planes = 0).std.Deflate(planes = 0).std.Deflate(planes = 0)
    mask = core.std.Minimum(mask, coordinates = [0, 0, 0, 1, 1, 0, 0, 0], planes = 0)
    mask = mt_binarize(core.std.Maximum(mask, planes = 0), thr1, planes = 0).std.Maximum(planes = 0)
    
    match mode:
        case 0:
            filt = vinverse(clip, 2.3, planes = planes)
        case 1:
            filt = vinverse2(clip, 2.3, planes = planes)
        case 2:
            filt = daa(clip, planes = planes, nns = 4, qual = 2, pscrn = 4, exp = 2)
        case _:
            raise ValueError(f'{func_name}: Please use 0...2 "mode" value')
    
    filt = core.std.MaskedMerge(clip, filt, mask, planes = planes, first_plane = True)
    
    clip = core.akarin.Select([clip, filt], core.std.PlaneStats(mask), f'x.PlaneStatsAverage {thr2 / 256} > 1 0 ?')
    
    return clip

def vinverse(clip: VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'vinverse'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    Vblur = core.std.Convolution(clip, [50, 99, 50], mode = 'v', planes = planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes = planes)
    
    expr0 = f'x x y - {sstr} * +'
    Vshrp = core.std.Convolution(Vblur, [1, 4, 6, 4, 1], mode = 'v', planes = planes)
    Vshrp = core.std.Expr([Vblur, Vshrp], [expr0 if i in planes else '' for i in range(num_p)])
    VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes = planes)
    
    expr1 = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD = core.std.Expr([VshrpD, VblurD], [expr1 if i in planes else '' for i in range(num_p)])
    
    result = core.std.MergeDiff(Vblur, VlimD, planes = planes)
    
    if amnt > 254:
        clip = result
    elif amnt == 0:
        pass
    else:
        amnt *= factor
        expr2 = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
        clip = core.std.Expr([clip, result], [expr2 if i in planes else '' for i in range(num_p)])
    
    return clip

def vinverse2(clip: VideoNode, sstr: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'vinverse2'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    Vblur = sbrV(clip, planes = planes)
    VblurD = core.std.MakeDiff(clip, Vblur, planes = planes)
    
    expr0 = f'x x y - {sstr} * +'
    Vshrp = core.std.Convolution(Vblur, [1, 2, 1], mode = 'v', planes = planes)
    Vshrp  = core.std.Expr([Vblur, Vshrp], [expr0 if i in planes else '' for i in range(num_p)])
    VshrpD = core.std.MakeDiff(Vshrp, Vblur, planes = planes)
    
    expr1 = f'x {half} - y {half} - * 0 < x {half} - abs y {half} - abs < x y ? {half} - {scl} * {half} + x {half} - abs y {half} - abs < x y ? ?'
    VlimD  = core.std.Expr([VshrpD, VblurD], [expr1 if i in planes else '' for i in range(num_p)])
    
    result = core.std.MergeDiff(Vblur, VlimD, planes = planes)
    
    if amnt > 254:
        clip = result
    elif amnt == 0:
        pass
    else:
        amnt *= factor
        expr2 = f'x {amnt} + y < x {amnt} + x {amnt} - y > x {amnt} - y ? ?'
        clip = core.std.Expr([clip, result], [expr2 if i in planes else '' for i in range(num_p)])
    
    return clip

def sbr(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'sbr'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    rg11 = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes = planes)
    rg11D = core.std.MakeDiff(clip, rg11, planes = planes)
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    rg11DD = core.std.Convolution(rg11D, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes = planes)
    rg11DD = core.std.MakeDiff(rg11D, rg11DD, planes = planes)
    rg11DD = core.std.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, rg11DD, planes = planes)
    
    return clip

def sbrV(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'sbrV'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    rg11 = core.std.Convolution(clip, [1, 2, 1], mode = 'v', planes = planes)
    rg11D = core.std.MakeDiff(clip, rg11, planes = planes)
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    rg11DD = core.std.Convolution(rg11D, [1, 2, 1], mode = 'v', planes = planes)
    rg11DD = core.std.MakeDiff(rg11D, rg11DD, planes = planes)
    rg11DD = core.std.Expr([rg11DD, rg11D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, rg11DD, planes = planes)
    
    return clip

def avs_Blur(clip: VideoNode, amountH: float = 0, amountV: float | None = None, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'avs_Blur'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    if amountV is None:
        amountV = amountH
    
    if amountH < -1 or amountV < -1 or amountH > 1.58 or amountV > 1.58:
        raise ValueError(f'{func_name}: the "amount" allowable range is from -1.0 to +1.58 ')
    
    center_h = 1 / 2 ** amountH
    side_h = (1 - 1 / 2 ** amountH) / 2
    
    center_v = 1 / 2 ** amountV
    side_v = (1 - 1 / 2 ** amountV) / 2
    
    if amountH:
        expr0 = f'x[-1,0] x[1,0] + {side_h} * x {center_h} * +'
        clip = core.akarin.Expr(clip, [expr0 if i in planes else '' for i in range(num_p)])
    
    if amountV:
        expr1 = f'x[0,-1] x[0,1] + {side_v} * x {center_v} * +'
        clip = core.akarin.Expr(clip, [expr1 if i in planes else '' for i in range(num_p)])
    
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
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.id != bright_limit.format.id or clip.format.id != dark_limit.format.id:
        raise ValueError(f'{func_name}: "clip", "bright_limit" and "dark_limit" must have the same format')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    overshoot *= factor
    undershoot *= factor
    
    expr = f'x z {undershoot} - y {overshoot} + clamp'
    clip = core.akarin.Expr([clip, bright_limit, dark_limit], [expr if i in planes else '' for i in range(num_p)])
    
    return clip

def MinBlur(clip: VideoNode, r: int = 1, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'MinBlur'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    half = 128 << clip.format.bits_per_sample - 8
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    match r:
        case 1:
            RG11D = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes = planes)
            RG11D = core.std.MakeDiff(clip, RG11D, planes = planes)
            
            RG4D = core.std.Median(clip, planes = planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes = planes)
        case 2:
            RG11D = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes = planes)
            RG11D = core.std.Convolution(RG11D, [1, 1, 1, 1, 1, 1, 1, 1, 1], planes = planes)
            RG11D = core.std.MakeDiff(clip, RG11D, planes = planes)
            
            RG4D = core.ctmf.CTMF(clip, 2, planes = planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes = planes)
        case 3:
            RG11D = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1], planes = planes)
            RG11D = core.std.Convolution(RG11D, [1, 1, 1, 1, 1, 1, 1, 1, 1], planes = planes)
            RG11D = core.std.Convolution(RG11D, [1, 1, 1, 1, 1, 1, 1, 1, 1], planes = planes)
            RG11D = core.std.MakeDiff(clip, RG11D, planes = planes)
            
            RG4D = core.ctmf.CTMF(clip, 3, planes = planes)
            RG4D = core.std.MakeDiff(clip, RG4D, planes = planes)
        case _:
            raise ValueError(f'{func_name}: Please use 1...3 "r" value')
    
    expr = f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?'
    DD = core.std.Expr([RG11D, RG4D], [expr if i in planes else '' for i in range(num_p)])
    
    clip = core.std.MakeDiff(clip, DD, planes = planes)
    
    return clip

def Dither_Luma_Rebuild(clip: VideoNode, s0: float = 2.0, c: float = 0.0625, planes: int | list[int] | None = None) -> VideoNode:
    
    func_name = 'Dither_Luma_Rebuild'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    factor = 1 << clip.format.bits_per_sample - 8
    half = 128 * factor
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
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
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    if thr_arg and (len(thr_arg) > 1 or 'threshold' not in thr_arg):
        raise ValueError(f'{func_name}: "thr_arg" must be "threshold" = float')
    
    if sw > 0 and sh > 0:
        if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1):
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
        clip = core.std.Maximum(clip, planes = planes, coordinates = mode_m, **thr_arg)
        clip = mt_expand_multi(clip, mode = mode, sw = sw - 1, sh = sh - 1, planes = planes, **thr_arg)
    
    return clip

def mt_inpand_multi(clip: VideoNode, mode: str = 'rectangle', sw: int = 1, sh: int = 1, planes: int | list[int] | None = None,
                    **thr_arg: float) -> VideoNode:
    
    func_name = 'mt_inpand_multi'
    
    num_p = clip.format.num_planes
    
    match planes:
        case None:
            planes = [*range(num_p)]
        case int():
            planes = [planes]
        case list() if all(isinstance(i, int) for i in planes):
            if len(planes) > num_p:
                raise ValueError(f'{func_name}: "planes" length must not be greater than the number of planes"')
        case _:
            raise ValueError(f'{func_name}: "planes" must be "int", "list[int]" or "None"')
    
    if thr_arg and (len(thr_arg) > 1 or 'threshold' not in thr_arg):
        raise ValueError(f'{func_name}: "thr_arg" must be "threshold" = float')
    
    if sw > 0 and sh > 0:
        if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1):
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
        clip = core.std.Minimum(clip, planes = planes, coordinates = mode_m, **thr_arg)
        clip = mt_inpand_multi(clip, mode = mode, sw = sw - 1, sh = sh - 1, planes = planes, **thr_arg)
    
    return clip
