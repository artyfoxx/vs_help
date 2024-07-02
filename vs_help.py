from vapoursynth import core, GRAY, YUV, VideoNode, VideoFrame, INTEGER
from muvsfunc import Blur, haf_Clamp, haf_MinBlur, sbr, rescale, haf_DitherLumaRebuild, haf_mt_expand_multi, haf_mt_inpand_multi
from typing import Any
from math import sqrt
from functools import partial
from inspect import signature

# All functions support the following formats: GRAY and YUV 8 - 16 bit integer. Floating point sample type is not supported.

# Functions:
# autotap3
# bion_dehalo
# fix_border
# mask_detail
# degrain_n
# destripe
# daa
# average_fields
# rg_fix
# znedi3aas
# dehalo_mask
# tp7_deband_mask
# dehalo_alpha
# fine_dehalo
# fine_dehalo2
# insane_aa
# upscaler
# custom_mask
# diff_mask
# apply_range
# titles_mask
# after_mask
# search_field_diffs

# Lanczos-based resize by "*.mp4 guy", ported from AviSynth version with minor additions.
# It is well suited for downsampling. Cropping parameters added in the form of **kwargs.

def autotap3(clip: VideoNode, dx: int | None = None, dy: int | None = None, mtaps3: int = 1, thresh: int = 256, **crop_args: float) -> VideoNode:
    
    func_name = 'autotap3'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w << 1
    if dy is None:
        dy = h << 1
    
    back_args = {}
    
    if len(crop_args) > 0:
        if 'src_left' in crop_args:
            back_args['src_left'] = -crop_args['src_left'] * dx / w
        
        if 'src_top' in crop_args:
            back_args['src_top'] = -crop_args['src_top'] * dy / h
        
        if 'src_width' in crop_args:
            if crop_args['src_width'] <= 0:
                crop_args['src_width'] += w - crop_args.get('src_left', 0)
            back_args['src_width'] = (dx << 1) - crop_args['src_width'] * dx / w
        
        if 'src_height' in crop_args:
            if crop_args['src_height'] <= 0:
                crop_args['src_height'] += h - crop_args.get('src_top', 0)
            back_args['src_height'] = (dy << 1) - crop_args['src_height'] * dy / h
        
        if not all((x := i) in back_args for i in crop_args):
            raise ValueError(f'{func_name}: Unsupported key {x} in crop_args')
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
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
    
    cp1 = core.std.MaskedMerge(Blur(t1, 1.42), t2, core.std.Expr([m1, m2], expr).resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
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


# Dehalo by bion, ported from AviSynth version with minor additions.
# mode = 1, 5, 11 - the weakest, artifacts will not cause.
# mode = 2, 3, 4 - bad modes, eat innocent parts, can't be used.
# mode = 10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
# mode = 14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.

def bion_dehalo(clip: VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1, m: bool = False) -> VideoNode:
    
    func_name = 'bion_dehalo'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    step = clip.format.bits_per_sample - 8
    half = 128 << step
    
    e1 = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], f'x y - {4 << step} - 4 *')
    e2 = core.std.Maximum(e1).std.Maximum()
    e2 = core.std.Merge(e2, core.std.Maximum(e2)).std.Inflate()
    e3 = core.std.Expr([core.std.Merge(e2, core.std.Maximum(e2)), core.std.Deflate(e1)], 'x y 1.2 * -').std.Inflate()
    
    m0 = core.std.Expr([clip, core.std.BoxBlur(clip, hradius = 2, vradius = 2)], 'x y - abs 0 > x y - 0.3125 * x + x ?')
    m1 = core.std.Expr([clip, m0], f'x y - {1 << step} - 128 *').std.Maximum().std.Inflate()
    m2 = core.std.Maximum(m1).std.Maximum()
    m3 = core.std.Expr([m1, m2], 'y x -').rgvs.RemoveGrain(21).std.Maximum()
    
    if mask == 1:
        pass
    elif mask == 2:
        e3 = m3
    elif mask == 3:
        e3 = core.std.Expr([e3, m3], 'x y min')
    elif mask == 4:
        e3 = core.std.Expr([e3, m3], 'x y max')
    else:
        raise ValueError(f'{func_name}: Please use 1...4 mask value')
    
    blurr = haf_MinBlur(clip, 1).std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if rg:
        dh1 = core.std.MaskedMerge(core.rgvs.Repair(clip, core.rgvs.RemoveGrain(clip, 21), 1), blurr, e3)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, e3)
    
    dh1D = core.std.MakeDiff(clip, dh1)
    tmp = sbr(dh1)
    med2D = core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2))
    DD  = core.std.Expr([dh1D, med2D], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs 2 * < x y {half} - 2 * {half} + ? ?')
    dh2 = core.std.MergeDiff(dh1, DD)
    
    clip = haf_Clamp(clip, core.rgvs.Repair(clip, dh2, mode) if rep else dh2, clip, 0, 20 << step)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if m:
        clip = e3 if space == GRAY else core.resize.Point(e3, format = orig.format.id)
    
    return clip


# A simple functions for fix brightness artifacts at the borders of the frame.
# All values are set as positional string arguments. The strings have the following format:
# [axis, target, donor, limit, mode, plane]. Only axis is mandatory.
# axis - can take the values "x" or "y" for columns and rows, respectively.
# target - the target column/row, it is counted from the upper left edge of the screen, by default 0.
# donor - the donor column/row, by default "None" (is calculated automatically as one closer to the center of the frame).
# limit - by default 0, without restrictions, positive values prohibit the darkening of target rows/columns
# and limit the maximum lightening, negative values - on the contrary, it's set in 8-bit notation.
# mode - target correction mode, by default 1, 0 - subtraction and addition, -1 and 1 - division and multiplication,
# -2 and 2 - logarithm and exponentiation, -3 and 3 - nth root and exponentiation.
# plane - by default 0.

def fix_border(clip: VideoNode, *args: list[str | int | None]) -> VideoNode:
    
    func_name = 'fix_border'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == GRAY:
        clips = [clip]
    elif space == YUV:
        num_p = clip.format.num_planes
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    for i in args:
        plane = i.pop() if len(i) == 6 else 0
        clips[plane] = eval(f'fix_border_{i[0]}_simple(clips[plane], *i[1:])')
    
    if space == GRAY:
        clip = clips[0]
    else:
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    
    return clip


def fix_border_x_simple(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, mode: int = 1) -> VideoNode:
    
    func_name = 'fix_border_x_simple'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family != GRAY:
        raise ValueError(f'{func_name}: Only GRAY is supported')
    
    limit *= 1 << clip.format.bits_per_sample - 8
    w = clip.width
    
    if donor is None:
        donor = target + 1 if target < w >> 1 else target - 1
    
    if mode == 0:
        expr = 'y.PlaneStatsAverage x.PlaneStatsAverage - x +'
    elif abs(mode) == 1:
        expr = 'y.PlaneStatsAverage x.PlaneStatsAverage / x *'
    elif abs(mode) == 2:
        expr = 'x y.PlaneStatsAverage log x.PlaneStatsAverage log / pow'
    elif abs(mode) == 3:
        expr = 'y.PlaneStatsAverage 1 x.PlaneStatsAverage / pow x pow'
    else:
        raise ValueError(f'{func_name}: Please use -3...3 mode value')
    
    if mode < 0:
        target_line = core.std.Crop(clip, target, w - target - 1, 0, 0).std.Invert().std.PlaneStats()
        donor_line = core.std.Crop(clip, donor, w - donor - 1, 0, 0).std.Invert().std.PlaneStats()
    else:
        target_line = core.std.Crop(clip, target, w - target - 1, 0, 0).std.PlaneStats()
        donor_line = core.std.Crop(clip, donor, w - donor - 1, 0, 0).std.PlaneStats()
    
    fix_line = core.akarin.Expr([target_line, donor_line], expr)
    
    if limit > 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y > x y x - {limit} < y x {limit} + ? ?')
    elif limit < 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y < x y x - {limit} > y x {limit} + ? ?')
    
    fix_line = core.std.RemoveFrameProps(fix_line, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    if mode < 0:
        fix_line = core.std.Invert(fix_line)
    
    if target == 0:
        clip = core.std.StackHorizontal([fix_line, core.std.Crop(clip, 1, 0, 0, 0)])
    elif target == w - 1:
        clip = core.std.StackHorizontal([core.std.Crop(clip, 0, 1, 0, 0), fix_line])
    else:
        clip = core.std.StackHorizontal([core.std.Crop(clip, 0, w - target, 0, 0), fix_line, core.std.Crop(clip, target + 1, 0, 0, 0)])
    
    return clip


def fix_border_y_simple(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, mode: int = 1) -> VideoNode:
    
    func_name = 'fix_border_y_simple'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family != GRAY:
        raise ValueError(f'{func_name}: Only GRAY is supported')
    
    limit *= 1 << clip.format.bits_per_sample - 8
    h = clip.height
    
    if donor is None:
        donor = target + 1 if target < h >> 1 else target - 1
    
    if mode == 0:
        expr = 'y.PlaneStatsAverage x.PlaneStatsAverage - x +'
    elif abs(mode) == 1:
        expr = 'y.PlaneStatsAverage x.PlaneStatsAverage / x *'
    elif abs(mode) == 2:
        expr = 'x y.PlaneStatsAverage log x.PlaneStatsAverage log / pow'
    elif abs(mode) == 3:
        expr = 'y.PlaneStatsAverage 1 x.PlaneStatsAverage / pow x pow'
    else:
        raise ValueError(f'{func_name}: Please use -3...3 mode value')
    
    if mode < 0:
        target_line = core.std.Crop(clip, 0, 0, target, h - target - 1).std.Invert().std.PlaneStats()
        donor_line = core.std.Crop(clip, 0, 0, donor, h - donor - 1).std.Invert().std.PlaneStats()
    else:
        target_line = core.std.Crop(clip, 0, 0, target, h - target - 1).std.PlaneStats()
        donor_line = core.std.Crop(clip, 0, 0, donor, h - donor - 1).std.PlaneStats()
    
    fix_line = core.akarin.Expr([target_line, donor_line], expr)
    
    if limit > 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y > x y x - {limit} < y x {limit} + ? ?')
    elif limit < 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y < x y x - {limit} > y x {limit} + ? ?')
    
    fix_line = core.std.RemoveFrameProps(fix_line, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    if mode < 0:
        fix_line = core.std.Invert(fix_line)
    
    if target == 0:
        clip = core.std.StackVertical([fix_line, core.std.Crop(clip, 0, 0, 1, 0)])
    elif target == h - 1:
        clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, 1), fix_line])
    else:
        clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, h - target), fix_line, core.std.Crop(clip, 0, 0, target + 1, 0)])
    
    return clip


# MaskDetail by "Tada no Snob", ported from AviSynth version with minor additions.
# Has nothing to do with the port by MonoS.
# It is based on the rescale class from muvsfunc, therefore it supports fractional resolutions
# and automatic width calculation based on the original aspect ratio.
# "down = True" is added for backward compatibility and does not support fractional resolutions.
# Also, this option is incompatible with using odd resolutions when there is chroma subsampling in the source.

def mask_detail(clip: VideoNode, dx: float | None = None, dy: float | None = None, rg: int = 3, cutoff: int = 70,
                gain: float = 0.75, blur_more: bool = False, kernel: str = 'bilinear', b: float = 0, c: float = 0.5,
                taps: int = 3, frac: bool = True, down: bool = False, **after_args: Any) -> VideoNode:
    
    func_name = 'mask_detail'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    step = clip.format.bits_per_sample - 8
    full = 256 << step
    w = clip.width
    h = clip.height
    
    if dy is None:
        dy = h // 2
    
    if kernel == 'bilinear':
        rescaler = rescale.Bilinear()
    elif kernel == 'bicubic':
        rescaler = rescale.Bicubic(b, c)
    elif kernel == 'lanczos':
        rescaler = rescale.Lanczos(taps)
    elif kernel == 'spline16':
        rescaler = rescale.Spline16()
    elif kernel == 'spline36':
        rescaler = rescale.Spline36()
    elif kernel == 'spline64':
        rescaler = rescale.Spline64()
    else:
        raise ValueError(f'{func_name}: {kernel} is unsupported kernel')
    
    if dx is None:
        resc = rescaler.rescale(clip, dy, h if frac else None)
    else:
        resc = rescaler.descale(clip, dx, dy, h if frac else None)
        resc = rescaler.upscale(resc, w, h)
    
    mask = core.std.MakeDiff(clip, resc).hist.Luma()
    mask = rg_fix_simple(mask, rg)
    mask = core.std.Expr(mask, f'x {cutoff << step} < 0 x {gain} {full} x + {full} / * * ?')
    
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
        
        mask = eval(f'core.resize.{kernel.capitalize()}(mask, dx, dy)')
    
    if blur_more:
        mask = core.std.Convolution(mask, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if space == YUV:
        mask = core.resize.Point(mask, format = format_id)
    
    return mask


# Just an alias for mv.Degrain
# The parameters of individual functions are set as dictionaries. Unloading takes place sequentially, separated by commas.
# If you do not set anything, the default settings of MVTools itself apply.
# Function dictionaries are set in order: Super, Analyze, Degrain, Recalculate.
# Recalculate is optional, but you can specify several of them (as many as you want).
# If you need to specify settings for only one function, the rest of the dictionaries are served empty.

def degrain_n(clip: VideoNode, *args: dict[str, Any], tr: int = 1, dark: bool = True) -> VideoNode:
    
    func_name = 'degrain_n'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if tr > 6 or tr < 1:
        raise ValueError(f'{func_name}: 1 <= "tr" <= 6')
    
    if len(args) < 3:
        args += ({},) * (3 - len(args))
    
    if dark:
        sup1 = haf_DitherLumaRebuild(clip, s0 = 1).mv.Super(**args[0])
        sup2 = core.mv.Super(clip, levels = 1, **args[0])
    else:
        sup1 = core.mv.Super(clip, **args[0])
    
    vectors = []
    
    for i in range(1, tr + 1):
        vectors.append(core.mv.Analyse(sup1, isb = True, delta = i, **args[1]))
        vectors.append(core.mv.Analyse(sup1, isb = False, delta = i, **args[1]))
    
    for i in args[3:]:
        for j in range(tr << 1):
            vectors[j] = core.mv.Recalculate(sup1, vectors[j], **i)
    
    clip = eval(f'core.mv.Degrain{tr}(clip, sup2 if dark else sup1, *vectors, **args[2])')
    
    return clip


# Simplified Destripe from YomikoR without any unnecessary conversions and soapy EdgeFixer
# The internal Descale functions are unloaded as usual.
# The function values that differ for the upper and lower fields are indicated in the list.

def destripe(clip: VideoNode, dx: int | None = None, dy: int | None = None, **descale_args: Any) -> VideoNode:
    
    func_name = 'destripe'
    
    if dx is None:
        dx = clip.width
    if dy is None:
        dy = clip.height >> 1
    
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
    
    clip = core.std.SeparateFields(clip, True)
    clip = core.std.SetFieldBased(clip, 0)
    
    clip_tf = core.descale.Descale(clip[::2], dx, dy, **descale_args)
    clip_bf = core.descale.Descale(clip[1::2], dx, dy, **second_args)
    
    clip = core.std.Interleave([clip_tf, clip_bf])
    clip = core.std.DoubleWeave(clip, True)[::2]
    clip = core.std.SetFieldBased(clip, 0)
    
    return clip


# daa by Didée, ported from AviSynth version with minor additions.

def daa(clip: VideoNode, planes: int | list[int] | None = None, **znedi3_args: Any) -> VideoNode:
    
    func_name = 'daa'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = [*range(num_p)]
    elif isinstance(planes, int):
        planes = [planes]
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [0.5 if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1] if clip.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]
    shrpD = core.std.MakeDiff(dbl, core.std.Convolution(dbl, matrix, planes = planes), planes = planes)
    DD = core.rgvs.Repair(shrpD, dblD, [13 if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip


# Just an experiment. It leads to a common denominator of the average normalized values of the fields of one frame.
# Ideally, it should fix interlaced fades painlessly, but in practice this does not always happen.
# Apparently it depends on the source.

def average_fields(clip: VideoNode, curve: int | list[int | None] | None = None, weight: float = 0.5, mode: int = 0) -> VideoNode:
    
    func_name = 'average_fields'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if curve is None:
        return clip
    elif isinstance(curve, int):
        curve = [curve] * num_p
    elif isinstance(curve, list):
        if len(curve) == num_p:
            pass
        elif len(curve) < num_p:
            curve += [None] * (num_p - len(curve))
        else:
            raise ValueError(f'{func_name}: "curve" must be shorter or the same length to number of planes, or "curve" must be "int"')
    else:
        raise ValueError(f'{func_name}: "curve" must be int, list or "None"')
    
    if space == GRAY:
        clip = average_fields_simple(clip, curve[0], weight, mode)
    elif space == YUV:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        
        for i in range(num_p):
            if curve[i] is not None:
                clips[i] = average_fields_simple(clips[i], curve[i], weight, mode)
        
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    return clip


def average_fields_simple(clip: VideoNode, curve: int | None = None, weight: float = 0.5, mode: int = 0) -> VideoNode:
    
    func_name = 'average_fields_simple'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if clip.format.color_family != GRAY:
        raise ValueError(f'{func_name}: Only GRAY is supported')
    
    if weight == 0:
        expr0 = 'x.PlaneStatsAverage'
    elif weight == 1:
        expr0 = 'y.PlaneStatsAverage'
    elif weight > 0 and weight < 1:
        expr0 = f'x.PlaneStatsAverage {1 - weight} * y.PlaneStatsAverage {weight} * +'
    else:
        raise ValueError(f'{func_name}: 0 <= "weight" <= 1')
    
    if curve is None:
        return clip
    elif curve == 0:
        expr1 = expr0 + ' x.PlaneStatsAverage - x +'
        expr2 = expr0 + ' y.PlaneStatsAverage - y +'
    elif abs(curve) == 1:
        expr1 = expr0 + ' x.PlaneStatsAverage / x *'
        expr2 = expr0 + ' y.PlaneStatsAverage / y *'
    elif abs(curve) == 2:
        expr1 = 'x ' + expr0 + ' log x.PlaneStatsAverage log / pow'
        expr2 = 'y ' + expr0 + ' log y.PlaneStatsAverage log / pow'
    elif abs(curve) == 3:
        expr1 = expr0 + ' 1 x.PlaneStatsAverage / pow x pow'
        expr2 = expr0 + ' 1 y.PlaneStatsAverage / pow y pow'
    else:
        raise ValueError(f'{func_name}: Please use -3...3 or "None" curve value')
    
    if curve < 0:
        clip = core.std.Invert(clip)
    
    if mode == 0:
        clip = core.std.SeparateFields(clip, True).std.PlaneStats()
        fields = [clip[::2], clip[1::2]]
        
        if weight == 0:
            fields[1] = core.akarin.Expr(fields, expr2)
        elif weight == 1:
            fields[0] = core.akarin.Expr(fields, expr1)
        else:
            fields[0], fields[1] = core.akarin.Expr(fields, expr1), core.akarin.Expr(fields, expr2)
        
        clip = core.std.Interleave(fields)
        clip = core.std.DoubleWeave(clip, True)[::2]
        clip = core.std.SetFieldBased(clip, 0)
    elif mode == 1:
        h = clip.height
        clips = [core.std.Crop(clip, 0, 0, i, h - i - 1).std.PlaneStats() for i in range(h)]
        
        if weight == 0:
            for i in range(1, h, 2):
                clips[i] = core.akarin.Expr([clips[i - 1], clips[i]], expr2)
        elif weight == 1:
            for i in range(0, h - 1, 2):
                clips[i] = core.akarin.Expr([clips[i], clips[i + 1]], expr1)
        else:
            for i in range(0, h - 1, 2):
                clips[i], clips[i + 1] = core.akarin.Expr([clips[i], clips[i + 1]], expr1), \
                                         core.akarin.Expr([clips[i], clips[i + 1]], expr2)
        
        clip = core.std.StackVertical(clips)
    elif mode == 2:
        clip = core.std.SeparateFields(clip, True)
        fields = [clip[::2], clip[1::2]]
        
        if weight == 0:
            fields[1] = fields[0]
        elif weight == 1:
            fields[0] = fields[1]
        else:
            fields[0] = fields[1] = core.std.Expr(fields, f'x {1 - weight} * y {weight} * +')
        
        clip = core.std.Interleave(fields)
        clip = core.std.DoubleWeave(clip, True)[::2]
        clip = core.std.SetFieldBased(clip, 0)
    else:
        raise ValueError(f'{func_name}: Please use 0...2 mode value')
    
    if curve < 0:
        clip = core.std.Invert(clip)
    
    clip = core.std.RemoveFrameProps(clip, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    return clip


# Alias for RemoveGrain. For internal use.

def rg_fix(clip: VideoNode, mode: int | list[int] = 2) -> VideoNode:
    
    func_name = 'rg_fix'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if isinstance(mode, int):
        mode = [mode] * num_p
    elif num_p == len(mode):
        pass
    elif num_p > len(mode):
        mode += [mode[-1]] * (num_p - len(mode))
    else:
        raise ValueError(f'{func_name}: "mode" must be shorter or the same length to number of planes, or "mode" must be "int"')
    
    if space == GRAY:
        clip = rg_fix_simple(clip, mode[0])
    elif space == YUV:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        for i in range(num_p):
            if mode[i]:
                clips[i] = rg_fix_simple(clips[i], mode[i])
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    return clip


def rg_fix_simple(clip: VideoNode, mode: int = 2) -> VideoNode:
    
    func_name = 'rg_fix_simple'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if mode == 0:
        pass
    elif mode == 4:
        clip = core.std.Median(clip)
    elif mode == 11 or mode == 12:
        clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif mode == 19:
        clip = core.std.Convolution(clip, [1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif mode == 20:
        clip = core.std.Convolution(clip, [1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        clip = core.rgvs.RemoveGrain(clip, mode)
    
    return clip


# nnedi2aas by Didée, ported from AviSynth version with minor additions.

def znedi3aas(clip: VideoNode, rg: int = 20, rep: int = 13, clamp: int = 0, planes: int | list[int] | None = None, **znedi3_args: Any) -> VideoNode:
    
    func_name = 'znedi3aas'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = [*range(num_p)]
    elif isinstance(planes, int):
        planes = [planes]
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [0.5 if i in planes else 0 for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    
    if clamp > 0:
        shrpD = core.std.MakeDiff(dbl, haf_Clamp(dbl, rg_fix(dbl, [rg if i in planes else 0 for i in range(num_p)]),
                                  dbl, 0, clamp << clip.format.bits_per_sample - 8, planes = planes), planes = planes)
    else:
        shrpD = core.std.MakeDiff(dbl, rg_fix(dbl, [rg if i in planes else 0 for i in range(num_p)]), planes = planes)
    
    DD = core.rgvs.Repair(shrpD, dblD, [rep if i in planes else 0 for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip


# Fork of jvsfunc.dehalo_mask from dnjulek with minor additions.
# Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.
# :param src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
# :param expand: Expansion of edge mask.
# :param iterations: Protects parallel lines and corners that are usually damaged by YAHR.
# :param brz: Adjusts the internal line thickness.
# :param shift: Corrective shift for fine-tuning iterations

def dehalo_mask(clip: VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> VideoNode:
    
    func_name = 'dehalo_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if brz > 255 or brz < 0:
        raise ValueError(f'{func_name}: brz must be between 0 and 255')

    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    step = clip.format.bits_per_sample - 8
    
    clip = core.std.Expr([clip, core.std.Maximum(clip).std.Maximum()], f'y x - {shift * (1 << step)} - 128 *')
    mask = core.tcanny.TCanny(clip, sigma = sqrt(expand * 2), mode = -1).std.Expr('x 16 *')
    
    for _ in range(iterations):
        clip = core.std.Maximum(clip)
    
    for _ in range(iterations):
        clip = core.std.Minimum(clip)
    
    clip = core.std.InvertMask(clip).std.BinarizeMask(80 << step)
    
    if brz < 255:
        clip = core.std.Inflate(clip).std.Inflate().std.BinarizeMask(brz << step)
    
    clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    mask = core.std.Expr([mask, clip], 'x y min')
    
    if space == YUV:
        mask = core.resize.Point(mask, format = format_id)
    
    return mask


def tp7_deband_mask(clip: VideoNode, thr: float | list[float] = 8, scale: float = 1, rg: bool = True, fake_prewitt: bool = False,
                    **after_args: Any) -> VideoNode:
    
    func_name = 'tp7_deband_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    bits = clip.format.bits_per_sample
    factor = 1 << bits - 8
    
    if fake_prewitt:
        clip = custom_mask(clip, 1, scale)
    else:
        clip = core.std.Prewitt(clip, scale = scale)
    
    if isinstance(thr, list):
        if num_p < len(thr):
            raise ValueError(f'{func_name}: "thr" must be shorter or the same length to number of planes, or "thr" must be "float"')
        
        clip = core.std.BinarizeMask(clip, [i * factor for i in thr])
    else:
        clip = core.std.BinarizeMask(clip, thr * factor)
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if space == GRAY:
        pass
    elif space == YUV:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        w = clip.width
        h = clip.height
        
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        clip = core.std.Expr(clips[1:], 'x y max')
        
        if sub_w > 0 or sub_h > 0:
            clip = core.fmtc.resample(clip, w, h, kernel = 'spline', taps = 6)
            if bits != 16:
                clip = core.fmtc.bitdepth(clip, bits = bits, dmode = 1)
        
        clip = core.std.Expr([clip, clips[0]], 'x y max')
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip


def dehalo_alpha(clip: VideoNode, rx: float = 2.0, ry: float = 2.0, darkstr: float = 1.0, brightstr: float = 1.0,
                 lowsens: float = 50, highsens: float = 50, ss: float = 1.5, showmask: bool = False) -> VideoNode:
    
    func_name = 'dehalo_alpha'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    w = clip.width
    h = clip.height
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    step = clip.format.bits_per_sample - 8
    full = 256 << step
    factor = 1 << step
    m4 = lambda x: 16 if x < 16 else int(x / 4 + 0.5) * 4
    
    halos = core.resize.Bicubic(clip, m4(w / rx), m4(h / ry), filter_param_a = 1/3, filter_param_b = 1/3).resize.Bicubic(w, h, filter_param_a = 1, filter_param_b = 0)
    are = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], 'x y -')
    ugly = core.std.Expr([core.std.Maximum(halos), core.std.Minimum(halos)], 'x y -')
    so = core.std.Expr([ugly, are], f'y x - y {0.001 * factor} + / {full - 1} * {lowsens * factor} - y {full} + {full << 1} / {highsens / 100} + *')
    lets = core.std.MaskedMerge(halos, clip, so)
    
    if ss == 1.0:
        remove = core.rgvs.Repair(clip, lets, 1)
    else:
        remove = core.resize.Lanczos(clip, m4(w * ss), m4(h * ss), filter_param_a = 3)
        remove = core.std.Expr([remove, core.std.Maximum(lets).resize.Bicubic(m4(w * ss), m4(h * ss), filter_param_a = 1/3, filter_param_b = 1/3)], 'x y min')
        remove = core.std.Expr([remove, core.std.Minimum(lets).resize.Bicubic(m4(w * ss), m4(h * ss), filter_param_a = 1/3, filter_param_b = 1/3)], 'x y max')
        remove = core.resize.Lanczos(remove, w, h, filter_param_a = 3)
    
    clip = core.std.Expr([clip, remove], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if showmask:
        clip = so if space == GRAY else core.resize.Point(so, format = orig.format.id)
    
    return clip


def fine_dehalo(clip: VideoNode, rx: float = 2, ry: float | None = None, thmi: int = 80, thma: int = 128, thlimi: int = 50,
                thlima: int = 100, darkstr: float = 1.0, brightstr: float = 1.0, lowsens: float = 50, highsens: float = 50,
                ss: float = 1.25, showmask: int = 0, contra: float = 0.0, excl: bool = True, edgeproc: float = 0.0, fake_prewitt = False) -> VideoNode:
    
    func_name = 'fine_dehalo'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
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
    
    dehaloed = dehalo_alpha(clip, rx, ry, darkstr, brightstr, lowsens, highsens, ss)
    
    if contra > 0:
        dehaloed = fine_dehalo_contrasharp(dehaloed, clip, contra)
    
    if fake_prewitt:
        edges = custom_mask(clip, 1)
    else:
        edges = core.std.Prewitt(clip)
    
    strong = core.std.Expr(edges, f'x {thmi} - {thma - thmi} / {full} *')
    large = haf_mt_expand_multi(strong, sw = rx_i, sh = ry_i)
    light = core.std.Expr(edges, f'x {thlimi} - {thlima - thlimi} / {full} *')
    shrink = haf_mt_expand_multi(light, mode = 'ellipse', sw = rx_i, sh = ry_i).std.Expr('x 4 *')
    shrink = haf_mt_inpand_multi(shrink, mode = 'ellipse', sw = rx_i, sh = ry_i)
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
            clip = outside if space == GRAY else core.resize.Point(outside, format = orig.format.id)
        elif showmask == 2:
            clip = shrink if space == GRAY else core.resize.Point(shrink, format = orig.format.id)
        elif showmask == 3:
            clip = edges if space == GRAY else core.resize.Point(edges, format = orig.format.id)
        elif showmask == 4:
            clip = strong if space == GRAY else core.resize.Point(strong, format = orig.format.id)
        else:
            raise ValueError(f'{func_name}: Please use 0...4 showmask value')
    
    return clip


def fine_dehalo_contrasharp(dehaloed: VideoNode, clip: VideoNode, level: float) -> VideoNode:
    
    func_name = 'fine_dehalo_contrasharp'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    half = 128 << dehaloed.format.bits_per_sample - 8
    
    bb = core.std.Convolution(dehaloed, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    bb2 = core.rgvs.Repair(bb, core.rgvs.Repair(bb, core.ctmf.CTMF(bb, 2), 1), 1)
    xd = core.std.MakeDiff(bb, bb2).std.Expr(f'x {half} - 2.49 * {level} * {half} +')
    xdd = core.std.Expr([xd, core.std.MakeDiff(clip, dehaloed)], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?')
    clip = core.std.MergeDiff(dehaloed, xdd)
    
    return clip


def fine_dehalo2(clip: VideoNode, hconv: list[int] | None = None, vconv: list[int] | None = None, showmask: bool = False) -> VideoNode:
    
    func_name = 'fine_dehalo2'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    
    if space == GRAY:
        pass
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    if hconv is None:
        hconv = [-1, -2, 0, 0, 40, 0, 0, -2, -1]
    
    if vconv is None:
        vconv = [-2, -1, 0, 0, 40, 0, 0, -1, -2]
    
    fix_h = core.std.Convolution(clip, vconv, mode = 'v')
    fix_v = core.std.Convolution(clip, hconv, mode = 'h')
    mask_h = core.std.Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], divisor = 4, saturate = False)
    mask_v = core.std.Convolution(clip, [1, 0, -1, 2, 0, -2, 1, 0, -1], divisor = 4, saturate = False)
    temp_h = core.std.Expr([mask_h, mask_v], 'x 3 * y -')
    temp_v = core.std.Expr([mask_v, mask_h], 'x 3 * y -')
    
    mask_h = fine_dehalo2_grow_mask(temp_h, 'v')
    mask_v = fine_dehalo2_grow_mask(temp_v, 'h')
    
    clip = core.std.MaskedMerge(clip, fix_h, mask_h)
    clip = core.std.MaskedMerge(clip, fix_v, mask_v)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    if showmask:
        clip = core.std.Expr([mask_h, mask_v], 'x y max')
        if space == YUV:
            clip = core.resize.Point(clip, format = orig.format.id)
    
    return clip


def fine_dehalo2_grow_mask(clip: VideoNode, mode: str) -> VideoNode:
    
    func_name = 'fine_dehalo2_grow_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if mode == 'v':
        coord = [0, 1, 0, 0, 0, 0, 1, 0]
    elif mode == 'h':
        coord = [0, 0, 0, 1, 1, 0, 0, 0]
    else:
        raise ValueError(f'{func_name}: {mode} is wrong mode')
    
    clip = core.std.Maximum(clip, coordinates = coord).std.Minimum(coordinates = coord)
    mask_1 = core.std.Maximum(clip, coordinates = coord)
    mask_2 = core.std.Maximum(mask_1, coordinates = coord).std.Maximum(coordinates = coord)
    clip = core.std.Expr([mask_2, mask_1], 'x y -')
    clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Expr('x 1.8 *')
    
    return clip


def insane_aa(clip: VideoNode, ext_aa: VideoNode = None, ext_mask: VideoNode = None, order: int = 0, mode: int = 0,
              desc_str: float = 0.3, kernel: str = 'bilinear', b: float = 1/3, c: float = 1/3, taps: int = 3,
              dx: int = None, dy: int = 720, dehalo: bool = False, masked: bool = False, frac: bool = True, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'insane_aa'
    
    space = clip.format.color_family
    
    if space == GRAY:
        orig_gray = clip
    elif space == YUV:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
        orig_gray = clip
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    if external_aa is None:
        w = clip.width
        h = clip.height
        
        if kernel == 'bilinear':
            rescaler = rescale.Bilinear()
        elif kernel == 'bicubic':
            rescaler = rescale.Bicubic(b, c)
        elif kernel == 'lanczos':
            rescaler = rescale.Lanczos(taps)
        elif kernel == 'spline16':
            rescaler = rescale.Spline16()
        elif kernel == 'spline36':
            rescaler = rescale.Spline36()
        elif kernel == 'spline64':
            rescaler = rescale.Spline64()
        else:
            raise ValueError(f'{func_name}: {kernel} is unsupported kernel')
        
        if dx is None:
            dx = w / h * dy
        
        clip = rescaler.descale(clip, dx, dy, h if frac else None)
        
        kwargs = rescaler.descale_args.copy()
        clip_sp = core.resize.Spline36(clip, **kwargs)
        
        clip = core.std.Merge(clip_sp, clip, desc_str)
        
        if dehalo:
            clip = fine_dehalo(clip, thmi = 45, thlimi = 60, thlima = 120, fake_prewitt = True)
        
        upscaler_mod = partial(upscaler, mode = mode, order = order, **upscaler_args)
        clip = rescaler.upscale(clip, w, h, upscaler_mod)
    else:
        if ext_aa.format.color_family == GRAY:
            clip = ext_aa
        else:
            raise ValueError(f'{func_name}: The external AA should be GRAY')
    
    if masked:
        if ext_mask is None:
            mask = core.std.Sobel(orig_gray, scale = 2).std.Maximum()
        else:
            if ext_mask.format.color_family == GRAY:
                mask = ext_mask
            else:
                raise ValueError(f'{func_name}: The external mask should be GRAY')
        
        clip = core.std.MaskedMerge(orig_gray, clip, mask)
    
    if space == YUV:
        clip = core.std.ShufflePlanes([clip, orig], [*range(orig.format.num_planes)], space)
    
    return clip


def upscaler(clip: VideoNode, dx: int | None = None, dy: int | None = None, src_left: float | None = None, src_top: float | None = None,
            src_width: float | None = None, src_height: float | None = None, mode: int = 0, order: int = 0, **upscaler_args: Any) -> VideoNode:
    
    func_name = 'upscaler'
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w
    if dy is None:
        dy = h
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
    
    if mode == 0:
        kernel = upscaler_args.pop('kernel', 'bicubic').capitalize()
        clip = eval(f'core.resize.{kernel}(clip, dx, dy, src_left = src_left, src_top = src_top, src_width = src_width, src_height = src_height, **upscaler_args)')
    elif mode in {1, 2, 3}:
        if order == 0:
            clip = edi3_aa(clip, mode, True, **upscaler_args)
        elif order == 1:
            clip = edi3_aa(clip, mode, False, **upscaler_args)
        elif order == 2:
            clip = core.std.Expr([edi3_aa(clip, mode, True, **upscaler_args), edi3_aa(clip, mode, False, **upscaler_args)], 'x y max')
        else:
            raise ValueError(f'{func_name}: Please use 0...2 order value')
        
        clip = autotap3(clip, dx, dy, src_left = src_left * 2 - 0.5, src_top = src_top * 2 - 0.5, src_width = src_width * 2, src_height = src_height * 2)
    else:
        raise ValueError(f'{func_name}: Please use 0...3 mode value')
    
    return clip


def edi3_aa(clip: VideoNode, mode: int = 1, order: bool = True, **edi3_args: Any) -> VideoNode:
    
    func_name = 'edi3_aa'
    
    if order:
        clip = core.std.Transpose(clip)
    
    if mode == 1:
        clip = core.znedi3.nnedi3(clip, field = 1, dh = True, **edi3_args)
        clip = core.std.Transpose(clip)
        clip = core.znedi3.nnedi3(clip, field = 1, dh = True, **edi3_args)
    elif mode == 2:
        clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, **edi3_args)
        clip = core.std.Transpose(clip)
        clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, **edi3_args)
    elif mode == 3:
        eedi3_args = {i:edi3_args[i] for i in signature(core.eedi3m.EEDI3).parameters if i in edi3_args}
        znedi3_args = {i:edi3_args[i] for i in signature(core.znedi3.nnedi3).parameters if i in edi3_args}
        
        if not all((x := i) in eedi3_args or x in znedi3_args for i in edi3_args):
            raise ValueError(f'{func_name}: Unsupported key {x} in edi3_args')
        
        clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, sclip = core.znedi3.nnedi3(clip, field = 1, dh = True, **znedi3_args), **eedi3_args)
        clip = core.std.Transpose(clip)
        clip = core.eedi3m.EEDI3(clip, field = 1, dh = True, sclip = core.znedi3.nnedi3(clip, field = 1, dh = True, **znedi3_args), **eedi3_args)
    else:
        raise ValueError(f'{func_name}: Please use 1...3 mode value')
    
    if not order:
        clip = core.std.Transpose(clip)
    
    return clip


def custom_mask(clip: VideoNode, mask: int = 0, scale: float = 1.0, boost: bool = False, offset: float = 0.0,
                **after_args: Any) -> VideoNode:
    
    func_name = 'custom_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if mask == 0:
        pass
    elif mask == 1:
        clip = core.std.Expr([core.std.Convolution(clip, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor = 1, saturate = False),
                              core.std.Convolution(clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor = 1, saturate = False),
                              core.std.Convolution(clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor = 1, saturate = False),
                              core.std.Convolution(clip, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor = 1, saturate = False)],
                              f'x y max z a max max {scale} *')
    elif mask == 2:
        clip = core.std.Expr([core.std.Convolution(clip, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor = 4, saturate = False),
                              core.std.Convolution(clip, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor = 4, saturate = False)],
                              f'x y max {scale} *')
    elif mask == 3:
        clip = core.std.Expr([core.std.Convolution(clip, [8, 16, 8, 0, 0, 0, -8, -16, -8], divisor = 8, saturate = False),
                              core.std.Convolution(clip, [8, 0, -8, 16, 0, -16, 8, 0, -8], divisor = 8, saturate = False)],
                              f'x y max {scale} *')
    else:
        raise ValueError(f'{func_name}: Please use 0...3 mask value')
    
    if boost:
        step = clip.format.bits_per_sample - 8
        clip = core.std.Expr(clip, f'x {128 << step} / 0.86 {offset} + pow {(256 << step) - 1} *')
    
    if len(after_args) > 0:
        clip = after_mask(clip, **after_args)
    
    return clip


def diff_mask(first: VideoNode, second: VideoNode, thr: float = 8, scale: float = 1.0, rg: bool = True,
              flatten: int = 0, **after_args: Any) -> VideoNode:
    
    func_name = 'diff_mask'
    
    if first.format.sample_type != INTEGER or second.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    if first.num_frames != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    space_f = first.format.color_family
    space_s = second.format.color_family
    
    if space_f == GRAY:
        pass
    elif space_f == YUV:
        format_id = first.format.id
        first = core.std.ShufflePlanes(first, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the first clip')
    
    if space_s == GRAY:
        pass
    elif space_s == YUV:
        second = core.std.ShufflePlanes(second, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family in the second clip')
    
    if (bits := first.format.bits_per_sample) == second.format.bits_per_sample:
        thr *= 1 << bits - 8
    else:
        raise ValueError(f'{func_name}: Sample types of clips do not match')
    
    clip = core.std.Expr([first, second], f'x y - abs {scale} *')
    
    if thr > 0:
        clip = core.std.BinarizeMask(clip, thr)
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if flatten > 0:
        for i in range(1, flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], 'x y max z max')
    elif flatten < 0:
        for i in range(1, -flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], 'x y min z min')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip


def apply_range(first: VideoNode, second: VideoNode, *args: int | list[int]) -> VideoNode:
    
    func_name = 'apply_range'
    
    if first.num_frames != second.num_frames:
        raise ValueError(f'{func_name}: The numbers of frames in the clips do not match')
    
    if first.format.name != second.format.name:
        raise ValueError(f'{func_name}: The clip formats do not match')
    
    for i in args:
        if isinstance(i, int):
            i = [i]
        elif isinstance(i, list):
            pass
        else:
            raise ValueError(f'{func_name}: *args must be list[int] or int')
        
        if len(i) == 2:
            if i[0] == 0:
                first = second[:i[1] + 1] + first[i[1] + 1:]
            elif i[1] == first.num_frames - 1:
                first = first[:i[0]] + second[i[0]:]
            elif i[0] < i[1]:
                first = first[:i[0]] + second[i[0]:i[1] + 1] + first[i[1] + 1:]
            else:
                raise ValueError(f'{func_name}: *args must be list[first_frame, last_frame] or int')
        elif len(i) == 1:
            if i[0] == 0:
                first = second[i[0]] + first[i[0] + 1:]
            elif i[0] == first.num_frames - 1:
                first = first[:i[0]] + second[i[0]]
            else:
                first = first[:i[0]] + second[i[0]] + first[i[0] + 1:]
        else:
            raise ValueError(f'{func_name}: *args length must be 1, 2 or must be "int"')
    
    return first


def titles_mask(clip: VideoNode, thr: float = 230, rg: bool = True, flatten: int = 0, **after_args: Any) -> VideoNode:
    
    func_name = 'titles_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    space = clip.format.color_family
    step = clip.format.bits_per_sample - 8
    
    if space == GRAY:
        pass
    elif space == YUV:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    else:
        raise ValueError(f'{func_name}: Unsupported color family')
    
    clip = core.std.BinarizeMask(clip, thr * (1 << step))
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if flatten > 0:
        for i in range(1, flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], 'x y max z max')
    elif flatten < 0:
        for i in range(1, -flatten + 1):
            clip = core.std.Expr([clip, clip[i:] + clip[-1] * i, clip[0] * i + clip[:-i]], 'x y min z min')
    
    if 'exp_n' not in after_args:
        after_args['exp_n'] = 1
    
    clip = after_mask(clip, **after_args)
    
    if space_f == YUV:
        clip = core.resize.Point(clip, format = format_id)
    
    return clip


def after_mask(clip: VideoNode, borders: list[int] | None = None, planes: int | list[int] | None = None,
                **after_args: int) -> VideoNode:
    
    func_name = 'after_mask'
    
    if clip.format.sample_type != INTEGER:
        raise ValueError(f'{func_name}: floating point sample type is not supported')
    
    num_p = clip.format.num_planes
    step = clip.format.bits_per_sample - 8
    
    if planes is None:
        planes = [*range(num_p)]
    elif isinstance(planes, int):
        planes = [planes]
    
    sample = dict(exp_n = 'Maximum', inp_n = 'Minimum', def_n = 'Deflate', inf_n = 'Inflate')
    
    for i in after_args:
        if i in sample:
            for _ in range(after_args[i]):
                clip = eval(f'core.std.{sample[i]}(clip, planes = planes)')
        else:
            raise ValueError(f'{func_name}: Unsupported key {i} in after_args')
    
    if borders is not None:
        if len(borders) == 4:
            pass
        elif len(borders) < 4:
            sample = [0, clip.width - 1, 0, clip.height - 1]
            borders += sample[len(borders):]
        else:
            raise ValueError(f'{func_name}: borders length must be <= 4')
        
        expr = f'X {borders[0]} >= X {borders[1]} <= Y {borders[2]} >= Y {borders[3]} <= and and and {(256 << step) - 1} 0 ? x min'
        clip = core.akarin.Expr(clip, [expr if i in planes else '' for i in range(num_p)])
    
    return clip


def search_field_diffs(clip: VideoNode, thr: float = 0.001, align: float | None = None, mode: int = 0,
                       output: str | None = None, plane: int = 0) -> VideoNode:
    
    func_name = 'search_field_diffs'
    
    if mode == 0:
        return clip
    elif abs(mode) in {1, 2, 3}:
        pass
    else:
        raise ValueError(f'{func_name}: Please use -3...3 mode value')
    
    if align is None:
        align = thr / 2
    
    if output is None:
        output = f'field_diffs_mode_{mode}_thr_{thr:.0e}.txt'
    
    num_f = clip.num_frames
    field_diffs = [0] * num_f
    
    def dump_diffs(n: int, f: list[VideoFrame], clip: VideoNode) -> VideoNode:
        
        field_diffs[n] = abs(f[0].props['PlaneStatsAverage'] - f[1].props['PlaneStatsAverage']) if mode > 0 else f[0].props['PlaneStatsDiff']
        
        if n == num_f - 1:
            with open(output, 'w', encoding = "UTF-8") as file:
                for i in range(num_f):
                    if abs(mode) == 1:
                        result = field_diffs[i]
                    elif abs(mode) == 2:
                        result = abs(field_diffs[i - 1 if i > 0 else 0] - field_diffs[i])
                    else:
                        result = abs(field_diffs[i - 1 if i > 0 else 0] - field_diffs[i]) if abs(field_diffs[i - 1 if i > 0 else 0] - field_diffs[i + 1 if i < num_f - 1 else num_f - 1]) <= align else 0
                    
                    if result >= thr:
                        file.write(f'{i} {result:.20f}\n')
        
        return clip
    
    temp = core.std.SeparateFields(clip, True)
    top = temp[::2]
    bottom = temp[1::2]
    top = core.std.PlaneStats(top, bottom, plane = plane)
    bottom = core.std.PlaneStats(bottom, plane = plane)
    
    clip = core.std.FrameEval(clip, partial(dump_diffs, clip = clip), prop_src = [top, bottom])
    
    return clip
