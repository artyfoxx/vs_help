from vapoursynth import core, GRAY, VideoNode
from muvsfunc import Blur, haf_Clamp, haf_MinBlur, sbr, haf_mt_expand_multi, haf_mt_inflate_multi, haf_mt_deflate_multi, rescale, haf_DitherLumaRebuild
from itertools import chain
from typing import Any


# Lanczos-based resize by "*.mp4 guy", ported from AviSynth version with minor additions.
# It is well suited for downsampling. Compared to the original, in addition to the porting itself,
# the entire internal reuse has been transferred to fmtconv. Crop parameters with fmtconv-style names have also been added.

def autotap3(clip: VideoNode, dx: int | None = None, dy: int | None = None, sx: float | None = None, sy: float | None = None,
             sw: float | None = None, sh: float | None = None, mtaps3: int = 1, thresh: int = 256) -> VideoNode:
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w << 1
    if dy is None:
        dy = h << 1
    
    if sx is None:
        sx = 0
        isx = 0
    else:
        isx = -sx * dx / w
    
    if sy is None:
        sy = 0
        isy = 0
    else:
        isy = -sy * dy / h
    
    if sw is None:
        sw = w
        isw = dx
    elif sw <= 0:
        sw = w - sx + sw
        isw = (dx << 1) - sw * dx / w
    else:
        isw = (dx << 1) - sw * dx / w
    
    if sh is None:
        sh = h
        ish = dy
    elif sh <= 0:
        sh = h - sy + sh
        ish = (dy << 1) - sh * dy / h
    else:
        ish = (dy << 1) - sh * dy / h
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
    bits = clip.format.bits_per_sample
    if bits < 16:
        clip = core.fmtc.bitdepth(clip, bits = 16)
    
    t1 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 1)
    t2 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 2)
    t3 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 3)
    t4 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 4)
    t5 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 5)
    t6 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 9)
    t7 = core.fmtc.resample(clip, dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = 36)
    
    m1 = core.std.Expr([clip, core.fmtc.resample(t1, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 1)], 'x y - abs')
    m2 = core.std.Expr([clip, core.fmtc.resample(t2, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 1)], 'x y - abs')
    m3 = core.std.Expr([clip, core.fmtc.resample(t3, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 1)], 'x y - abs')
    m4 = core.std.Expr([clip, core.fmtc.resample(t4, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 2)], 'x y - abs')
    m5 = core.std.Expr([clip, core.fmtc.resample(t5, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 2)], 'x y - abs')
    m6 = core.std.Expr([clip, core.fmtc.resample(t6, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 3)], 'x y - abs')
    m7 = core.std.Expr([clip, core.fmtc.resample(t7, w, h, isx, isy, isw, ish, kernel = "lanczos", taps = 6)], 'x y - abs')
    
    cp1 = core.std.MaskedMerge(Blur(t1, amountH = 1.42), t2, core.fmtc.resample(core.std.Expr([m1, m2], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    m100 = core.std.Expr([clip, core.fmtc.resample(cp1, w, h, isx, isy, isw, ish, kernel = "bilinear")], 'x y - abs')
    cp2 = core.std.MaskedMerge(cp1, t3, core.fmtc.resample(core.std.Expr([m100, m3], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    m101 = core.std.Expr([clip, core.fmtc.resample(cp2, w, h, isx, isy, isw, ish, kernel = "bilinear")], 'x y - abs')
    cp3 = core.std.MaskedMerge(cp2, t4, core.fmtc.resample(core.std.Expr([m101, m4], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    m102 = core.std.Expr([clip, core.fmtc.resample(cp3, w, h, isx, isy, isw, ish, kernel = "bilinear")], 'x y - abs')
    cp4 = core.std.MaskedMerge(cp3, t5, core.fmtc.resample(core.std.Expr([m102, m5], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    m103 = core.std.Expr([clip, core.fmtc.resample(cp4, w, h, isx, isy, isw, ish, kernel = "bilinear")], 'x y - abs')
    cp5 = core.std.MaskedMerge(cp4, t6, core.fmtc.resample(core.std.Expr([m103, m6], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    m104 = core.std.Expr([clip, core.fmtc.resample(cp5, w, h, isx, isy, isw, ish, kernel = "bilinear")], 'x y - abs')
    clip = core.std.MaskedMerge(cp5, t7, core.fmtc.resample(core.std.Expr([m104, m7], f'x y - {thresh} *'), dx, dy, sx, sy, sw, sh, kernel = "lanczos", taps = mtaps3))
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, core.fmtc.resample(orig, dx, dy, sx, sy, sw, sh, kernel = "spline36")], list(range(orig.format.num_planes)), space)
    
    if bits < 16:
        clip = core.fmtc.bitdepth(clip, bits = bits)
    
    return clip


# Dehalo by bion-x, ported from AviSynth version with minor additions.
# Supported formats: YUV 8 - 16 bit integer.
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
    e2 = haf_mt_expand_multi(e1, sw = 2, sh = 2)
    e2 = core.std.Merge(e2, haf_mt_expand_multi(e2))
    e2 = haf_mt_inflate_multi(e2)
    e3 = core.std.Merge(e2, haf_mt_expand_multi(e2))
    e3 = core.std.Expr([e3, haf_mt_deflate_multi(e1)], 'x y 1.2 * -')
    e3 = haf_mt_inflate_multi(e3)
    
    m0 = core.std.BoxBlur(clip, hradius = 2, vradius = 2)
    m0 = core.std.Expr([clip, m0], 'x y - abs 0 > x y - 0.3125 * x + x ?')
    m1 = core.std.Expr([clip, m0], f'x y - {1 << step} - 128 *')
    m1 = haf_mt_expand_multi(m1)
    m1 = haf_mt_inflate_multi(m1)
    m2 = haf_mt_expand_multi(m1, sw = 2, sh = 2)
    m3 = core.std.Expr([m1, m2], 'y x -')
    m3 = core.rgvs.RemoveGrain(m3, 21)
    m3 = haf_mt_expand_multi(m3)
    
    if mask == 1:
        pass
    elif mask == 2:
        e3 = m3
    elif mask == 3:
        e3 = core.std.Expr([e3, m3], 'x y min')
    elif mask == 4:
        e3 = core.std.Expr([e3, m3], 'x y max')
    else:
        raise ValueError('dehalo: Please use 1...4 mask type')
    
    blurr = haf_MinBlur(clip, 1)
    blurr = core.std.Convolution(blurr, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    blurr = core.std.Convolution(blurr, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if rg:
        dh1 = core.rgvs.Repair(clip, core.rgvs.RemoveGrain(clip, 21), 1)
        dh1 = core.std.MaskedMerge(dh1, blurr, e3)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, e3)
    
    dh1D = core.std.MakeDiff(clip, dh1)
    tmp = sbr(dh1)
    med2D = core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2))
    DD  = core.std.Expr([dh1D, med2D], f'x {half} - y {half} - * 0 < {half} x {half} - abs y {half} - abs 2 * < x y {half} - 2 * {half} + ? ?')
    dh2 = core.std.MergeDiff(dh1, DD)
    
    r = core.rgvs.Repair(clip, dh2, mode) if rep else dh2
    
    clip = haf_Clamp(clip, r, clip, 0, 20 << step)
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, orig], list(range(orig.format.num_planes)), space)
    
    if m:
        clip = e3 if space == GRAY else core.resize.Point(clip, format = orig.format.id)
    
    return clip


# Custom upscaler for the rescale class from muvsfunc. Just a hardline znedi3 upscale with autotap3.

def znedi3at(clip: VideoNode, dx: int | None = None, dy: int | None = None, sx: float | None = None, sy: float | None = None,
             sw: float | None = None, sh: float | None = None) -> VideoNode:
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w
    if dy is None:
        dy = h
    if sx is None:
        sx = 0
    if sy is None:
        sy = 0
    if sw is None:
        sw = w
    elif sw <= 0:
        sw = w - sx + sw
    if sh is None:
        sh = h
    elif sh <= 0:
        sh = h - sy + sh
    
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = autotap3(clip, dx, dy, sx * 2 - 0.5, sy * 2 - 0.5, sw * 2, sh * 2)
    
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

def FixBorder(clip: VideoNode, tx: int | list[int] | None = None, ty: int | list[int] | None = None, dx: int | list[int | None] | None = None,
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
            raise ValueError('FixBorder: "dx" must be shorter or the same length to "tx", or "dx" must be "int" or "None"')
        
        if isinstance(lx, int):
            lx = [lx] * length_x
        elif lx is None:
            lx = [0] * length_x
        elif length_x == len(lx):
            pass
        elif length_x > len(lx):
            lx += [lx[-1]] * (length_x - len(lx))
        else:
            raise ValueError('FixBorder: "lx" must be shorter or the same length to "tx", or "lx" must be "int" or "None"')
        
        if isinstance(px, int):
            px = [px] * length_x
        elif px is None:
            px = [0] * length_x
        elif length_x == len(px):
            pass
        elif length_x > len(px):
            px += [px[-1]] * (length_x - len(px))
        else:
            raise ValueError('FixBorder: "px" must be shorter or the same length to "tx", or "px" must be "int" or "None"')
        
        for i in range(length_x):
            clip = FixBorderX(clip, tx[i], dx[i], lx[i], px[i])
    
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
            raise ValueError('FixBorder: "dy" must be shorter or the same length to "ty", or "dy" must be "int" or "None"')
        
        if isinstance(ly, int):
            ly = [ly] * length_y
        elif ly is None:
            ly = [0] * length_y
        elif length_y == len(ly):
            pass
        elif length_y > len(ly):
            ly += [ly[-1]] * (length_y - len(ly))
        else:
            raise ValueError('FixBorder: "ly" must be shorter or the same length to "ty", or "ly" must be "int" or "None"')
        
        if isinstance(py, int):
            py = [py] * length_y
        elif py is None:
            py = [0] * length_y
        elif length_y == len(py):
            pass
        elif length_y > len(py):
            py += [py[-1]] * (length_y - len(py))
        else:
            raise ValueError('FixBorder: "py" must be shorter or the same length to "ty", or "py" must be "int" or "None"')
        
        for i in range(length_y):
            clip = FixBorderY(clip, ty[i], dy[i], ly[i], py[i])
    
    return clip

def FixBorderX(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
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

def FixBorderY(clip: VideoNode, target: int = 0, donor: int | None = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
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

def MaskDetail(clip: VideoNode, dx: float | None = None, dy: float | None = None, RGmode: int = 3, cutoff: int = 70,
               gain: float = 0.75, expandN: int = 2, inflateN: int = 1, blur_more: bool = False, kernel: str = 'bilinear',
               b: float = 0, c: float = 0.5, taps: int = 3, frac: bool = True, down: bool = False, **down_args: Any) -> VideoNode:
    
    if dy is None:
        raise ValueError('MaskDetail: "dy" must be specified')
    
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
        raise ValueError('MaskDetail: Unsupported kernel type')
    
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
    
    diff = core.std.MakeDiff(clip, resc)
    initial_mask = core.hist.Luma(diff)
    initial_mask = RemoveGrainFix(initial_mask, RGmode)
    initial_mask = core.std.Expr(initial_mask, f'x {cutoff << step} < 0 x {gain} {full} x + {full} / * * ?')
    expanded = haf_mt_expand_multi(initial_mask, sw = expandN, sh = expandN)
    final = haf_mt_inflate_multi(expanded, radius = inflateN)
    
    if down:
        if dx is None:
            raise ValueError('MaskDetail: if "down" is "True", then "dx" can\'t be "None"')
        
        if not isinstance(dx, int) or not isinstance(dy, int):
            raise ValueError('MaskDetail: if "down" is "True", then "dx" and "dy" must be "int"')
        
        if space != GRAY and (dx >> sub_w << sub_w != dx or dy >> sub_h << sub_h != dy):
            raise ValueError('MaskDetail: "dx" or "dy" does not match the chroma subsampling of the output clip')
        
        final = core.resize.Bilinear(final, dx, dy, **down_args)
    
    if blur_more:
        final = core.std.Convolution(final, [1, 2, 1, 2, 4, 2, 1, 2, 1])
    
    if space != GRAY:
        final = core.resize.Point(final, format = format_id)
    
    return final


# Just an alias for mv.Degrain
# The parameters of individual functions are set as dictionaries. Unloading takes place sequentially, separated by commas.
# If you do not set anything, the default settings of MVTools itself apply.
# Function dictionaries are set in order: Super, Analyze, Degrain, Recalculate.
# Recalculate is optional, but you can specify several of them (as many as you want).
# If you need to specify settings for only one function, the rest of the dictionaries are served empty.

def MDegrainN(clip: VideoNode, tr: int = 1, *args: dict[str, Any], dark: bool = True) -> VideoNode:
    
    if tr > 6 or tr < 1:
        raise ValueError('MDegrainN: 1 <= "tr" <= 6')
    
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
        mvbw = [core.mv.Recalculate(sup1, mvbw[j], **i) for j in range(tr)]
        mvfw = [core.mv.Recalculate(sup1, mvfw[j], **i) for j in range(tr)]
    
    clip = eval(f'core.mv.Degrain{tr}(clip, sup2 if dark else sup1, *chain.from_iterable(zip(mvbw, mvfw)), **args[2])')
    
    return clip


# Simplified Destripe from YomikoR without any unnecessary conversions and soapy EdgeFixer
# The internal Descale functions are unloaded as a dictionary.
# The function values that differ for the upper and lower fields are indicated in the list.

def Destripe(clip: VideoNode, dx: int | None = None, dy: int | None = None, **descale_args: Any) -> VideoNode:
    
    if dx is None:
        dx = clip.width
    if dy is None:
        dy = clip.height >> 1
    
    descale_args2 = {}
    for i in descale_args:
        if isinstance(descale_args[i], list):
            descale_args2[i] = descale_args[i][1]
            descale_args[i] = descale_args[i][0]
        else:
            descale_args2[i] = descale_args[i]
    
    clip = core.std.SeparateFields(clip, True)
    clip = core.std.SetFieldBased(clip, 0)
    
    clip_tf = clip[::2].descale.Descale(dx, dy, **descale_args)
    clip_bf = clip[1::2].descale.Descale(dx, dy, **descale_args2)
    
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

def averagefields(clip: VideoNode, planes: int | list[int] | None = None) -> VideoNode:
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if planes is None:
        planes = list(range(num_p))
    elif isinstance(planes, int):
        planes = [planes]
    
    clip = core.std.SeparateFields(clip, True)
    
    for i in planes:
        if i >= num_p:
            raise ValueError(f'averagefields: plane {i} does not exist')
        
        if space != GRAY:
            orig = clip
            clip = core.std.ShufflePlanes(clip, i, GRAY)
        
        clip = core.std.PlaneStats(clip)
        fields = [clip[::2], clip[1::2]]
        clip_tf = core.akarin.Expr(fields, 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / x.PlaneStatsAverage / x *')
        clip_bf = core.akarin.Expr(fields, 'x.PlaneStatsAverage y.PlaneStatsAverage + 2 / y.PlaneStatsAverage / y *')
        clip = core.std.Interleave([clip_tf, clip_bf])
        clip = core.std.RemoveFrameProps(clip, ['PlaneStatsMin', 'PlaneStatsMax', 'PlaneStatsAverage'])
        
        if space != GRAY:
            clip = core.std.ShufflePlanes([(clip if i == j else orig) for j in range(num_p)], [(0 if i == j else j) for j in range(num_p)], space)
    
    clip = core.std.DoubleWeave(clip, True)[::2]
    clip = core.std.SetFieldBased(clip, 0)
    
    return clip


# Alias for RemoveGrain. For internal use.

def RemoveGrainFix(clip: VideoNode, mode: int | list[int] = 2) -> VideoNode:
    
    space = clip.format.color_family
    num_p = clip.format.num_planes
    
    if isinstance(mode, int):
        mode = [mode] * num_p
    elif num_p > len(mode):
        mode += [mode[-1]] * (num_p - len(mode))
    elif num_p == len(mode):
        pass
    else:
        raise ValueError('RemoveGrainFix: "mode" must be shorter or the same length to number of planes, or "mode" must be "int"')
    
    count = 0
    
    for i in mode:
        if space != GRAY:
            orig = clip
            clip = core.std.ShufflePlanes(clip, count, GRAY)
        
        if i == 0:
            pass
        elif i == 4:
            clip = core.std.Median(clip)
        elif i == 11 or i == 12:
            clip = core.std.Convolution(clip, [1, 2, 1, 2, 4, 2, 1, 2, 1])
        elif i == 19:
            clip = core.std.Convolution(clip, [1, 1, 1, 1, 0, 1, 1, 1, 1])
        elif i == 20:
            clip = core.std.Convolution(clip, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        else:
            clip = core.rgvs.RemoveGrain(clip, i)
        
        if space != GRAY:
            clip = core.std.ShufflePlanes([(clip if count == j else orig) for j in range(num_p)], [(0 if count == j else j) for j in range(num_p)], space)
        
        count += 1
    
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
        shrpD = core.std.MakeDiff(dbl, haf_Clamp(dbl, RemoveGrainFix(dbl, [(rg if i in planes else 0) for i in range(num_p)]), dbl, 0, clamp << clip.format.bits_per_sample - 8, planes = planes), planes = planes)
    else:
        shrpD = core.std.MakeDiff(dbl, RemoveGrainFix(dbl, [(rg if i in planes else 0) for i in range(num_p)]), planes = planes)
    
    DD = core.rgvs.Repair(shrpD, dblD, [(rep if i in planes else 0) for i in range(num_p)])
    clip = core.std.MergeDiff(dbl, DD, planes = planes)
    
    return clip
