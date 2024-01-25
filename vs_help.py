from vapoursynth import core, GRAY, VideoNode
from muvsfunc import Blur, haf_Clamp, haf_MinBlur, sbr, rescale, haf_DitherLumaRebuild, haf_mt_expand_multi, haf_mt_inpand_multi
from itertools import chain
from typing import Any
from math import sqrt

# All filters support the following formats: YUV 8 - 16 bit integer. Float is not supported yet.


# Lanczos-based resize by "*.mp4 guy", ported from AviSynth version with minor additions.
# It is well suited for downsampling. Cropping parameters added in the form of **kwargs.

def autotap3(clip: VideoNode, dx: int | None = None, dy: int | None = None, mtaps3: int = 1, thresh: int = 256, **crop_args: float) -> VideoNode:
    
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
        
        if len(crop_args) != len(back_args):
            raise ValueError('autotap3: Unsupported keys in crop_args')
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
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
    
    cp1 = core.std.MaskedMerge(Blur(t1, 1.42), t2, core.std.Expr([m1, m2], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m100 = core.std.Expr([clip, core.resize.Bilinear(cp1, w, h, **back_args)], 'x y - abs')
    cp2 = core.std.MaskedMerge(cp1, t3, core.std.Expr([m100, m3], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m101 = core.std.Expr([clip, core.resize.Bilinear(cp2, w, h, **back_args)], 'x y - abs')
    cp3 = core.std.MaskedMerge(cp2, t4, core.std.Expr([m101, m4], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m102 = core.std.Expr([clip, core.resize.Bilinear(cp3, w, h, **back_args)], 'x y - abs')
    cp4 = core.std.MaskedMerge(cp3, t5, core.std.Expr([m102, m5], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m103 = core.std.Expr([clip, core.resize.Bilinear(cp4, w, h, **back_args)], 'x y - abs')
    cp5 = core.std.MaskedMerge(cp4, t6, core.std.Expr([m103, m6], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    m104 = core.std.Expr([clip, core.resize.Bilinear(cp5, w, h, **back_args)], 'x y - abs')
    clip = core.std.MaskedMerge(cp5, t7, core.std.Expr([m104, m7], f'x y - {thresh} *').resize.Lanczos(dx, dy, filter_param_a = mtaps3, **crop_args))
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, core.resize.Spline36(orig, dx, dy, **crop_args)], list(range(orig.format.num_planes)), space)
    
    return clip


# Dehalo by bion-x, ported from AviSynth version with minor additions.
# mode = 1, 5, 11 - the weakest, artifacts will not cause.
# mode = 2, 3, 4 - bad modes, eat innocent parts, can't be used.
# mode = 10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
# mode = 14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.

def dehalo(clip: VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1, m: bool = False) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
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
        raise ValueError('dehalo: Please use 1...4 mask value')
    
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
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if m:
        clip = e3 if space == GRAY else core.resize.Point(e3, format = orig.format.id)
    
    return clip


# Custom upscaler for the rescale class from muvsfunc. Just a hardline znedi3 upscale with autotap3.

def znedi3at(clip: VideoNode, dx: int | None = None, dy: int | None = None, src_left: float | None = None, src_top: float | None = None,
             src_width: float | None = None, src_height: float | None = None) -> VideoNode:
    
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
    
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = autotap3(clip, dx, dy, **dict(src_left = src_left * 2 - 0.5, src_top = src_top * 2 - 0.5, src_width = src_width * 2, src_height = src_height * 2))
    
    return clip


# A simple functions for fix brightness artifacts at the borders of the frame.
# The values of the target columns/rows are set as lists of tx and ty or as an integer.
# The values of the reference columns/rows are set as lists dx and dy or as an integer.
# You can also set it as "None" or specify nothing at all, in this case, auto mode is enabled,
# assigning the reference rows/columns as a single offset to the center of the frame relative to the target ones.
# Next are the limits lx and ly, which can also be both integers and lists of them. A single limit applies to all iterations on a given axis.
# Positive values prohibit darkening of the target rows/columns and limit the maximum lightening, negative values - on the contrary.
# By default, the limits are zero, that is, they are disabled.
# Last are plans px and py, which can also be both integers and lists of them. By default, the zero plan is set.

def fix_border(clip: VideoNode, tx: int | list[int] | None = None, ty: int | list[int] | None = None, dx: int | list[int | None] | None = None,
              dy: int | list[int | None] | None = None, lx: int | list[int] | None = None, ly: int | list[int] | None = None,
              px: int | list[int] | None = None, py: int | list[int] | None = None) -> VideoNode:
    
    if tx is not None:
        if isinstance(tx, int):
            tx = [tx]
        
        length_x = len(tx)
        
        if isinstance(dx, int):
            dx = [dx] + [None] * (length_x - 1)
        elif dx is None:
            dx = [None] * length_x
        elif length_x == len(dx):
            pass
        elif length_x > len(dx):
            dx += [None] * (length_x - len(dx))
        else:
            raise ValueError('fix_border: "dx" must be shorter or the same length to "tx", or "dx" must be "int" or "None"')
        
        if isinstance(lx, int):
            lx = [lx] * length_x
        elif lx is None:
            lx = [0] * length_x
        elif length_x == len(lx):
            pass
        elif length_x > len(lx):
            lx += [lx[-1]] * (length_x - len(lx))
        else:
            raise ValueError('fix_border: "lx" must be shorter or the same length to "tx", or "lx" must be "int" or "None"')
        
        if isinstance(px, int):
            px = [px] * length_x
        elif px is None:
            px = [0] * length_x
        elif length_x == len(px):
            pass
        elif length_x > len(px):
            px += [px[-1]] * (length_x - len(px))
        else:
            raise ValueError('fix_border: "px" must be shorter or the same length to "tx", or "px" must be "int" or "None"')
        
        for i in range(length_x):
            clip = fix_border_x(clip, tx[i], dx[i], lx[i], px[i])
    
    if ty is not None:
        if isinstance(ty, int):
            ty = [ty]
        
        length_y = len(ty)
        
        if isinstance(dy, int):
            dy = [dy] + [None] * (length_y - 1)
        elif dy is None:
            dy = [None] * length_y
        elif length_y == len(dy):
            pass
        elif length_y > len(dy):
            dy += [None] * (length_y - len(dy))
        else:
            raise ValueError('fix_border: "dy" must be shorter or the same length to "ty", or "dy" must be "int" or "None"')
        
        if isinstance(ly, int):
            ly = [ly] * length_y
        elif ly is None:
            ly = [0] * length_y
        elif length_y == len(ly):
            pass
        elif length_y > len(ly):
            ly += [ly[-1]] * (length_y - len(ly))
        else:
            raise ValueError('fix_border: "ly" must be shorter or the same length to "ty", or "ly" must be "int" or "None"')
        
        if isinstance(py, int):
            py = [py] * length_y
        elif py is None:
            py = [0] * length_y
        elif length_y == len(py):
            pass
        elif length_y > len(py):
            py += [py[-1]] * (length_y - len(py))
        else:
            raise ValueError('fix_border: "py" must be shorter or the same length to "ty", or "py" must be "int" or "None"')
        
        for i in range(length_y):
            clip = fix_border_y(clip, ty[i], dy[i], ly[i], py[i])
    
    return clip

def fix_border_x(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        num_p = clip.format.num_planes
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    w = clip.width
    
    if donor is None:
        donor = target + 1 if target < w >> 1 else target - 1
    
    target_line = core.std.Crop(clip, target, w - target - 1, 0, 0).std.PlaneStats()
    donor_line = core.std.Crop(clip, donor, w - donor - 1, 0, 0).std.PlaneStats()
    
    fix_line = core.akarin.Expr([target_line, donor_line], 'y.PlaneStatsAverage x.PlaneStatsAverage / x *')
    
    if limit > 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y > x y x - {limit} < y x {limit} + ? ?')
    elif limit < 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y < x y x - {limit} > y x {limit} + ? ?')
    
    fix_line = core.std.RemoveFrameProps(fix_line, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    if target == 0:
        clip = core.std.StackHorizontal([fix_line, core.std.Crop(clip, 1, 0, 0, 0)])
    elif target == w - 1:
        clip = core.std.StackHorizontal([core.std.Crop(clip, 0, 1, 0, 0), fix_line])
    else:
        clip = core.std.StackHorizontal([core.std.Crop(clip, 0, w - target, 0, 0), fix_line, core.std.Crop(clip, target + 1, 0, 0, 0)])
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([(clip if i == plane else orig) for i in range(num_p)], [(0 if i == plane else i) for i in range(num_p)], space)
    
    return clip

def fix_border_y(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        num_p = clip.format.num_planes
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    h = clip.height
    
    if donor is None:
        donor = target + 1 if target < h >> 1 else target - 1
    
    target_line = core.std.Crop(clip, 0, 0, target, h - target - 1).std.PlaneStats()
    donor_line = core.std.Crop(clip, 0, 0, donor, h - donor - 1).std.PlaneStats()
    
    fix_line = core.akarin.Expr([target_line, donor_line], 'y.PlaneStatsAverage x.PlaneStatsAverage / x *')
    
    if limit > 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y > x y x - {limit} < y x {limit} + ? ?')
    elif limit < 0:
        fix_line = core.std.Expr([target_line, fix_line], f'x y < x y x - {limit} > y x {limit} + ? ?')
    
    fix_line = core.std.RemoveFrameProps(fix_line, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    if target == 0:
        clip = core.std.StackVertical([fix_line, core.std.Crop(clip, 0, 0, 1, 0)])
    elif target == h - 1:
        clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, 1), fix_line])
    else:
        clip = core.std.StackVertical([core.std.Crop(clip, 0, 0, 0, h - target), fix_line, core.std.Crop(clip, 0, 0, target + 1, 0)])
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([(clip if i == plane else orig) for i in range(num_p)], [(0 if i == plane else i) for i in range(num_p)], space)
    
    return clip


# MaskDetail by "Tada no Snob", ported from AviSynth version with minor additions.
# Has nothing to do with the port by MonoS.
# It is based on the rescale class from muvsfunc, therefore it supports fractional resolutions
# and automatic width calculation based on the original aspect ratio.
# "down = True" is added for backward compatibility and does not support fractional resolutions.
# Also, this option is incompatible with using odd resolutions when there is chroma subsampling in the source.

def mask_detail(clip: VideoNode, dx: float | None = None, dy: float | None = None, rg: int = 3, cutoff: int = 70,
               gain: float = 0.75, expand_n: int = 2, inflate_n: int = 1, blur_more: bool = False, kernel: str = 'bilinear',
               b: float = 0, c: float = 0.5, taps: int = 3, frac: bool = True, down: bool = False, **down_args: Any) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        format_id = clip.format.id
        sub_w = clip.format.subsampling_w
        sub_h = clip.format.subsampling_h
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
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
        raise ValueError('mask_detail: Unsupported kernel type')
    
    if dx is None:
        if frac:
            resc = rescaler.rescale(clip, dy, h)
        else:
            resc = rescaler.rescale(clip, dy)
    else:
        if frac:
            desc = rescaler.descale(clip, dx, dy, h)
        else:
            desc = rescaler.descale(clip, dx, dy)
        resc = rescaler.upscale(desc, w, h)
    
    mask = core.std.MakeDiff(clip, resc).hist.Luma()
    mask = rg_fix_simple(mask, rg)
    mask = core.std.Expr(mask, f'x {cutoff << step} < 0 x {gain} {full} x + {full} / * * ?')
    
    for _ in range(expand_n):
        mask = core.std.Maximum(mask)
    
    for _ in range(inflate_n):
        mask = core.std.Inflate(mask)
    
    if down:
        if dx is None:
            raise ValueError('mask_detail: if "down" is "True", then "dx" can\'t be "None"')
        
        if not isinstance(dx, int) or not isinstance(dy, int):
            raise ValueError('mask_detail: if "down" is "True", then "dx" and "dy" must be "int"')
        
        if space != GRAY and (dx >> sub_w << sub_w != dx or dy >> sub_h << sub_h != dy):
            raise ValueError('mask_detail: "dx" or "dy" does not match the chroma subsampling of the output clip')
        
        mask = core.resize.Bilinear(mask, dx, dy, **down_args)
    
    if blur_more:
        mask = core.std.Convolution(mask, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if space != GRAY:
        mask = core.resize.Point(mask, format = format_id)
    
    return mask


# Just an alias for mv.Degrain
# The parameters of individual functions are set as dictionaries. Unloading takes place sequentially, separated by commas.
# If you do not set anything, the default settings of MVTools itself apply.
# Function dictionaries are set in order: Super, Analyze, Degrain, Recalculate.
# Recalculate is optional, but you can specify several of them (as many as you want).
# If you need to specify settings for only one function, the rest of the dictionaries are served empty.

def degrain_n(clip: VideoNode, *args: dict[str, Any], tr: int = 1, dark: bool = True) -> VideoNode:
    
    if tr > 6 or tr < 1:
        raise ValueError('degrain_n: 1 <= "tr" <= 6')
    
    if len(args) < 3:
        args += ({},) * (3 - len(args))
    
    if dark:
        sup1 = haf_DitherLumaRebuild(clip, s0 = 1).mv.Super(**args[0])
        sup2 = core.mv.Super(clip, levels = 1, **args[0])
    else:
        sup1 = core.mv.Super(clip, **args[0])
    
    mvbw = [core.mv.Analyse(sup1, isb = True, delta = i, **args[1]) for i in range(1, tr + 1)]
    mvfw = [core.mv.Analyse(sup1, isb = False, delta = i, **args[1]) for i in range(1, tr + 1)]
    
    for i in args[3:]:
        for j in range(tr):
            mvbw[j] = core.mv.Recalculate(sup1, mvbw[j], **i)
            mvfw[j] = core.mv.Recalculate(sup1, mvfw[j], **i)
    
    clip = eval(f'core.mv.Degrain{tr}(clip, sup2 if dark else sup1, *chain.from_iterable(zip(mvbw, mvfw)), **args[2])')
    
    return clip


# Simplified Destripe from YomikoR without any unnecessary conversions and soapy EdgeFixer
# The internal Descale functions are unloaded as a dictionary.
# The function values that differ for the upper and lower fields are indicated in the list.

def destripe(clip: VideoNode, dx: int | None = None, dy: int | None = None, **descale_args: Any) -> VideoNode:
    
    if dx is None:
        dx = clip.width
    if dy is None:
        dy = clip.height >> 1
    
    second_args = {}
    for i in descale_args:
        if isinstance(descale_args[i], list):
            second_args[i] = descale_args[i][1]
            descale_args[i] = descale_args[i][0]
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
    
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = list(range(num_p))
    elif isinstance(planes, int):
        planes = [planes]
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [(0.5 if i in planes else 0) for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1] if clip.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]
    shrpD = core.std.MakeDiff(dbl, core.std.Convolution(dbl, matrix, planes = planes), planes = planes)
    DD = core.rgvs.Repair(shrpD, dblD, [(13 if i in planes else 0) for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip


# Just an experiment. It leads to a common denominator of the average normalized values of the fields of one frame.
# Ideally, it should fix interlaced fades painlessly, but in practice this does not always happen.
# Apparently it depends on the source.

def average_fields(clip: VideoNode, mode: int = 0, planes: int | list[int] | None = None) -> VideoNode:
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = list(range(num_p))
    elif isinstance(planes, int):
        if planes in list(range(num_p)):
            planes = [planes]
        else:
            raise ValueError(f'average_fields: Plane {planes} not exist')
    elif isinstance(planes, list) and not set(planes).issubset(set(range(num_p))):
        raise ValueError('average_fields: Invalid plane in planes')
    else:
        raise ValueError('average_fields: "planes" must be "None", "int" or list')
    
    if space != GRAY:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        for i in range(num_p):
            if i in planes:
                clips[i] = average_fields_simple(clips[i], mode)
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    else:
        clip = average_fields_simple(clip, mode)
    
    return clip


def average_fields_simple(clip: VideoNode, mode: int = 0) -> VideoNode:
    
    if clip.format.color_family != GRAY:
        raise ValueError('average_fields_simple: Only GRAY is supported')
    
    if mode == 0:
        clip = core.std.SeparateFields(clip, True).std.PlaneStats()
        fields = [clip[::2], clip[1::2]]
        fields[0], fields[1] = (core.akarin.Expr(fields, 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / x.PlaneStatsAverage / x *'),
                                core.akarin.Expr(fields, 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / y.PlaneStatsAverage / y *'))
        clip = core.std.Interleave(fields)
        clip = core.std.DoubleWeave(clip, True)[::2]
        clip = core.std.SetFieldBased(clip, 0)
    elif mode == 1:
        h = clip.height
        clips = [core.std.Crop(clip, 0, 0, i, h - i - 1).std.PlaneStats() for i in range(h)]
        for i in range(0, h - 1, 2):
            clips[i], clips[i + 1] = (core.akarin.Expr([clips[i], clips[i + 1]], 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / x.PlaneStatsAverage / x *'),
                                      core.akarin.Expr([clips[i], clips[i + 1]], 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / y.PlaneStatsAverage / y *'))
        clip = core.std.StackVertical(clips)
    else:
        raise ValueError('average_fields_simple: Please use 0 or 1 mode value')
    
    clip = core.std.RemoveFrameProps(clip, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
    
    return clip


# Alias for RemoveGrain. For internal use.

def rg_fix(clip: VideoNode, mode: int | list[int] = 2) -> VideoNode:
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if isinstance(mode, int):
        clip = rg_fix_simple(clip, mode)
        return clip
    elif num_p == len(mode):
        pass
    elif num_p > len(mode):
        mode += [mode[-1]] * (num_p - len(mode))
    else:
        raise ValueError('rg_fix: "mode" must be shorter or the same length to number of planes, or "mode" must be "int"')
    
    if space != GRAY:
        clips = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        for i in range(num_p):
            clips[i] = rg_fix_simple(clips[i], mode[i])
        clip = core.std.ShufflePlanes(clips, [0] * num_p, space)
    else:
        clip = rg_fix_simple(clip, mode[0])
    
    return clip


def rg_fix_simple(clip: VideoNode, mode: int = 2) -> VideoNode:
    
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
    
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = list(range(num_p))
    elif isinstance(planes, int):
        planes = [planes]
    
    nn = core.znedi3.nnedi3(clip, field = 3, planes = planes, **znedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2], [(0.5 if i in planes else 0) for i in range(num_p)])
    
    dblD = core.std.MakeDiff(clip, dbl, planes = planes)
    
    if clamp > 0:
        shrpD = core.std.MakeDiff(dbl, haf_Clamp(dbl, rg_fix(dbl, [(rg if i in planes else 0) for i in range(num_p)]), dbl, 0, clamp << clip.format.bits_per_sample - 8, planes = planes), planes = planes)
    else:
        shrpD = core.std.MakeDiff(dbl, rg_fix(dbl, [(rg if i in planes else 0) for i in range(num_p)]), planes = planes)
    
    DD = core.rgvs.Repair(shrpD, dblD, [(rep if i in planes else 0) for i in range(num_p)])
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
    
    if brz > 255 or brz < 0:
        raise ValueError('dehalo_mask: brz must be between 0 and 255')

    space = clip.format.color_family
    if space != GRAY:
        format_id = clip.format.id
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
    step = clip.format.bits_per_sample - 8
    
    clip = core.std.Expr([clip, core.std.Maximum(clip).std.Maximum()], f'y x - {shift << step} - 128 *')
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
    
    if space != GRAY:
        mask = core.resize.Point(mask, format = format_id)
    
    return mask

def tp7_deband_mask(clip: VideoNode, thr: float | list[float] = 8, scale: float = 1, rg: bool = True, exp_n: int = 1) -> VideoNode:
    
    num_p = clip.format.num_planes
    bits = clip.format.bits_per_sample
    step = bits - 8
    
    if isinstance(thr, list):
        if num_p == len(thr):
            pass
        elif num_p > len(thr):
            thr += [thr[-1]] * (num_p - len(thr))
        else:
            raise ValueError('tp7_deband_mask: "thr" must be shorter or the same length to number of planes, or "thr" must be "float"')
        
        clip = core.std.Prewitt(clip, scale = scale).std.BinarizeMask([thr[i] * (1 << step) for i in range(num_p)])
    else:
        clip = core.std.Prewitt(clip, scale = scale).std.BinarizeMask(thr * (1 << step))
    
    if rg:
        clip = core.rgvs.RemoveGrain(clip, 3).std.Median()
    
    if clip.format.color_family != GRAY:
        format_id = clip.format.id
        
        mask = [core.std.ShufflePlanes(clip, i, GRAY) for i in range(num_p)]
        
        for i in range(num_p - 2, 0, -1):
            mask[i] = core.std.Expr([mask[i], mask[i + 1]], 'x y max')
        
        if clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0:
            mask[1] = core.fmtc.resample(mask[1], clip.width, clip.height, kernel = "spline", taps = 6)
            if bits != 16:
                mask[1] = core.fmtc.bitdepth(mask[1], bits = bits, dmode = 1)
        
        clip = core.std.Expr([mask[0], mask[1]], 'x y max')
        
        for _ in range(exp_n):
            clip = core.std.Maximum(clip)
        
        clip = core.resize.Point(clip, format = format_id)
    
    return clip


def dehalo_alpha(clip: VideoNode, rx: float = 2.0, ry: float = 2.0, darkstr: float = 1.0, brightstr: float = 1.0,
                 lowsens: float = 50, highsens: float = 50, ss: float = 1.5) -> VideoNode:
    
    w = clip.width
    h = clip.height
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
    step = clip.format.bits_per_sample - 8
    full = 256 << step
    mult = 1 << step
    m4 = lambda x: 16 if x < 16 else int(x / 4 + 0.5) * 4
    
    halos = core.resize.Bicubic(clip, m4(w / rx), m4(h / ry), filter_param_a = 1/3, filter_param_b = 1/3).resize.Bicubic(w, h, filter_param_a = 1, filter_param_b = 0)
    are = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], 'x y -')
    ugly = core.std.Expr([core.std.Maximum(halos), core.std.Minimum(halos)], 'x y -')
    so = core.std.Expr([ugly, are], f'y x - y {0.001 * mult} + / {full - 1} * {lowsens * mult} - y {full} + {512 << step} / {highsens / 100} + *')
    lets = core.std.MaskedMerge(halos, clip, so)
    
    if ss == 1.0:
        remove = core.rgvs.Repair(clip, lets, 1)
    else:
        remove = core.resize.Lanczos(clip, m4(w * ss), m4(h * ss), filter_param_a = 3)
        remove = core.std.Expr([remove, core.std.Maximum(lets).resize.Bicubic(m4(w * ss), m4(h * ss), filter_param_a = 1/3, filter_param_b = 1/3)], 'x y min')
        remove = core.std.Expr([remove, core.std.Minimum(lets).resize.Bicubic(m4(w * ss), m4(h * ss), filter_param_a = 1/3, filter_param_b = 1/3)], 'x y max')
        remove = core.resize.Lanczos(remove, w, h, filter_param_a = 3)
    
    clip = core.std.Expr([clip, remove], f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    return clip


def fine_dehalo(clip: VideoNode, rx: float = 2, ry: float | None = None, thmi: int = 80, thma: int = 128, thlimi: int = 50,
                thlima: int = 100, darkstr: float = 1.0, brightstr: float = 1.0, lowsens: float = 50, highsens: float = 50,
                ss: float = 1.5, showmask: int = 0, contra: float = 0.0, excl: bool = True, edgeproc: float = 0.0) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
    step = clip.format.bits_per_sample - 8
    thmi <<= step
    thma <<= step
    thlimi <<= step
    thlima <<= step
    full = (256 << step) - 1
    
    if ry is None:
        ry = rx
    
    rx_i = int(rx + 0.5)
    ry_i = int(ry + 0.5)
    
    dehaloed = dehalo_alpha(clip, rx, ry, darkstr, brightstr, lowsens, highsens, ss)
    
    if contra > 0:
        dehaloed = fine_dehalo_contrasharp(dehaloed, clip, contra)
    
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
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
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
            raise ValueError('fine_dehalo: Please use 0...4 showmask value')
    
    return clip


def fine_dehalo_contrasharp(dehaloed: VideoNode, clip: VideoNode, level: float) -> VideoNode:
    
    step = dehaloed.format.bits_per_sample - 8
    half = 128 << step
    
    bb = core.std.Convolution(dehaloed, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    bb2 = core.rgvs.Repair(bb, core.rgvs.Repair(bb, core.ctmf.CTMF(bb, 2), 1), 1)
    xd = core.std.MakeDiff(bb, bb2).std.Expr(f'x {half} - 2.49 * {level} * {half} +')
    xdd = core.std.Expr([xd, core.std.MakeDiff(clip, dehaloed)], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs < x y ? ?')
    clip = core.std.MergeDiff(dehaloed, xdd)
    
    return clip


def fine_dehalo2(clip: VideoNode, hconv: list[int] = [-1, -2, 0, 0, 40, 0, 0, -2, -1], vconv: list[int] = [-2, -1, 0, 0, 40, 0, 0, -1, -2],
                 showmask: bool = False) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
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
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if showmask:
        clip = core.std.Expr([mask_h, mask_v], 'x y max')
        if space != GRAY:
            clip = core.resize.Point(clip, format = orig.format.id)
    
    return clip


def fine_dehalo2_grow_mask(clip: VideoNode, mode: str) -> VideoNode:
    
    if mode == 'v':
        coord = [0, 1, 0, 0, 0, 0, 1, 0]
    elif mode == 'h':
        coord = [0, 0, 0, 1, 1, 0, 0, 0]
    else:
        raise ValueError('fine_dehalo2_grow_mask: wrong mode')
    
    clip = core.std.Maximum(clip, coordinates = coord).std.Minimum(coordinates = coord)
    mask_1 = core.std.Maximum(clip, coordinates = coord)
    mask_2 = core.std.Maximum(mask_1, coordinates = coord).std.Maximum(coordinates = coord)
    clip = core.std.Expr([mask_2, mask_1], 'x y -')
    clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Expr('x 1.8 *')
    
    return clip
