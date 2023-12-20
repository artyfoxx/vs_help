from vapoursynth import core, GRAY, VideoNode
from typing import Optional, Union, Sequence
from muvsfunc import Blur, haf_Clamp, haf_MinBlur, sbr, haf_mt_expand_multi, haf_mt_inflate_multi, haf_mt_deflate_multi, rescale

# Lanczos-based resize by "*.mp4 guy", ported from AviSynth version with minor additions and moved to fmtconv.
def autotap3(clip: VideoNode, dx: Optional[int] = None, dy: Optional[int] = None, mtaps3: int = 1, thresh: int = 256) -> VideoNode:
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w << 1
    
    if dy is None:
        dy = h << 1
    
    space = clip.format.color_family
    if space != GRAY:
        chroma = clip
        clip = core.std.ShufflePlanes(clip, 0, GRAY)
    
    bits = clip.format.bits_per_sample
    if bits < 16:
        clip = core.fmtc.bitdepth(clip, bits = 16)
    
    t1 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 1)
    t2 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 2)
    t3 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 3)
    t4 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 4)
    t5 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 5)
    t6 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 9)
    t7 = core.fmtc.resample(clip, dx, dy, kernel = "lanczos", taps = 36)
    
    m1 = core.std.Expr([clip, core.fmtc.resample(t1, w, h, kernel = "lanczos", taps = 1)], 'x y - abs')
    m2 = core.std.Expr([clip, core.fmtc.resample(t2, w, h, kernel = "lanczos", taps = 1)], 'x y - abs')
    m3 = core.std.Expr([clip, core.fmtc.resample(t3, w, h, kernel = "lanczos", taps = 1)], 'x y - abs')
    m4 = core.std.Expr([clip, core.fmtc.resample(t4, w, h, kernel = "lanczos", taps = 2)], 'x y - abs')
    m5 = core.std.Expr([clip, core.fmtc.resample(t5, w, h, kernel = "lanczos", taps = 2)], 'x y - abs')
    m6 = core.std.Expr([clip, core.fmtc.resample(t6, w, h, kernel = "lanczos", taps = 3)], 'x y - abs')
    m7 = core.std.Expr([clip, core.fmtc.resample(t7, w, h, kernel = "lanczos", taps = 6)], 'x y - abs')
    
    cp1 = core.std.MaskedMerge(Blur(t1, amountH = 1.42), t2, core.fmtc.resample(core.std.Expr([m1, m2], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    m100 = core.std.Expr([clip, core.fmtc.resample(cp1, w, h, kernel = "bilinear")], 'x y - abs')
    cp2 = core.std.MaskedMerge(cp1, t3, core.fmtc.resample(core.std.Expr([m100, m3], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    m101 = core.std.Expr([clip, core.fmtc.resample(cp2, w, h, kernel = "bilinear")], 'x y - abs')
    cp3 = core.std.MaskedMerge(cp2, t4, core.fmtc.resample(core.std.Expr([m101, m4], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    m102 = core.std.Expr([clip, core.fmtc.resample(cp3, w, h, kernel = "bilinear")], 'x y - abs')
    cp4 = core.std.MaskedMerge(cp3, t5, core.fmtc.resample(core.std.Expr([m102, m5], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    m103 = core.std.Expr([clip, core.fmtc.resample(cp4, w, h, kernel = "bilinear")], 'x y - abs')
    cp5 = core.std.MaskedMerge(cp4, t6, core.fmtc.resample(core.std.Expr([m103, m6], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    m104 = core.std.Expr([clip, core.fmtc.resample(cp5, w, h, kernel = "bilinear")], 'x y - abs')
    clip = core.std.MaskedMerge(cp5, t7, core.fmtc.resample(core.std.Expr([m104, m7], f'x y - {thresh} *'), dx, dy, kernel = "lanczos", taps = mtaps3))
    
    if space != GRAY:
        clip = core.std.ShufflePlanes([clip, core.fmtc.resample(chroma, dx, dy, kernel = "spline36")], list(range(chroma.format.num_planes)), chroma.format.color_family)
    
    if bits < 16:
        clip = core.fmtc.bitdepth(clip, bits = bits)
    
    return clip

# Dehalo by bion-x, ported from AviSynth version with minor additions.
def dehalo(clip: VideoNode, mode: int = 13, rep: bool = True, rg: bool = False, mask: int = 1, m: bool = False) -> VideoNode:
    
    '''
    Supported formats: YUV 8 - 16 bit integer.
    mode = 1, 5, 11 - the weakest, artifacts will not cause.
    mode = 2, 3, 4 - bad modes, eat innocent parts, can not be used.
    mode = 10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
    mode = 14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.
    '''
    
    space = clip.format.color_family
    if space != GRAY:
        chroma = clip
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
    blurr = core.rgvs.RemoveGrain(blurr, 11)
    blurr = core.rgvs.RemoveGrain(blurr, 11)
    
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
        clip = core.std.ShufflePlanes([clip, chroma], list(range(chroma.format.num_planes)), chroma.format.color_family)
    
    if m:
        clip = e3 if space == GRAY else core.resize.Point(clip, format = chroma.format.id)
    
    return clip

# Just a hardline znedi3 upscale with autotap3.
def znedi3at(clip: VideoNode, target_width: Optional[int] = None, target_height: Optional[int] = None, src_left: Optional[float] = None,
             src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None) -> VideoNode:
    
    if clip.format.color_family != GRAY:
        raise ValueError('znedi3at: Only \"GRAY\" clip is supported')
    
    w = clip.width
    h = clip.height
    
    if target_width is None:
        target_width = w
    if target_height is None:
        target_height = h
    if src_left is None:
        src_left = 0
    if src_top is None:
        src_top = 0
    if src_width is None:
        src_width = w
    elif src_width <= 0:
        src_width = w - src_left + src_width
    if src_height is None:
        src_height = h
    elif src_height <= 0:
        src_height = h - src_top + src_height
    
    bits = clip.format.bits_per_sample
    
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = core.std.Transpose(clip)
    clip = core.znedi3.nnedi3(clip, field = 1, dh = True, nsize = 0, nns = 4, qual = 2, pscrn = 0, exp = 2)
    clip = core.fmtc.resample(clip, kernel = "spline64", sx = src_left * 2 - 0.5, sy = src_top * 2 - 0.5, sw = src_width * 2, sh = src_height * 2)
    clip = autotap3(clip, target_width, target_height)
    
    if bits < 16:
        clip = core.fmtc.bitdepth(clip, bits = bits)
    
    return clip

# A simple function for fix brightness artifacts at the borders of the frame.
def FixBorder(clip: VideoNode, tx: Optional[Union[int, Sequence[int]]] = None, ty: Optional[Union[int, Sequence[int]]] = None,
              dx: Optional[Union[int, Sequence[int]]] = None, dy: Optional[Union[int, Sequence[int]]] = None,
              lx: Optional[Union[int, Sequence[int]]] = None, ly: Optional[Union[int, Sequence[int]]] = None,
              px: Optional[Union[int, Sequence[int]]] = None, py: Optional[Union[int, Sequence[int]]] = None) -> VideoNode:
    
    if tx is not None:
        if isinstance(tx, int):
            tx = [tx]
        if isinstance(dx, int):
            dx = [dx]
        elif dx is None:
            dx = tx
        length_x = len(tx)
        if length_x == len(dx):
            if isinstance(lx, int):
                lx = [lx for _ in range(length_x)]
            elif lx is None:
                lx = [0 for _ in range(length_x)]
            elif length_x > len(lx):
                for _ in range(length_x - len(lx)):
                    lx.append(lx[len(lx) - 1])
            if length_x == len(lx):
                if isinstance(px, int):
                    px = [px for _ in range(length_x)]
                elif px is None:
                    px = [0 for _ in range(length_x)]
                elif length_x > len(px):
                    for _ in range(length_x - len(px)):
                        px.append(px[len(px) - 1])
                if length_x == len(px):
                    for i in range(length_x):
                        clip = FixBorderX(clip, tx[i], dx[i], lx[i], px[i])
                else:
                    raise ValueError('FixBorder: \"px\" must be shorter or the same length to \"tx\", or \"px\" must be \"int\" or \"None\"')
            else:
                raise ValueError('FixBorder: \"lx\" must be shorter or the same length to \"tx\", or \"lx\" must be \"int\" or \"None\"')
        else:
            raise ValueError('FixBorder: \"dx\" must be the same length to \"tx\", or \"dx\" must be \"None\"')
    
    if ty is not None:
        if isinstance(ty, int):
            ty = [ty]
        if isinstance(dy, int):
            dy = [dy]
        elif dy is None:
            dy = ty
        length_y = len(ty)
        if length_y == len(dy):
            if isinstance(ly, int):
                ly = [ly for _ in range(length_y)]
            elif ly is None:
                ly = [0 for _ in range(length_y)]
            elif length_y > len(ly):
                for _ in range(length_y - len(ly)):
                    ly.append(ly[len(ly) - 1])
            if length_y == len(ly):
                if isinstance(py, int):
                    py = [py for _ in range(length_y)]
                elif py is None:
                    py = [0 for _ in range(length_y)]
                elif length_y > len(py):
                    for _ in range(length_y - len(py)):
                        py.append(py[len(py) - 1])
                if length_y == len(py):
                    for i in range(length_y):
                        clip = FixBorderY(clip, ty[i], dy[i], ly[i], py[i])
                else:
                    raise ValueError('FixBorder: \"py\" must be shorter or the same length to \"ty\", or \"py\" must be \"int\" or \"None\"')
            else:
                raise ValueError('FixBorder: \"ly\" must be shorter or the same length to \"ty\", or \"ly\" must be \"int\" or \"None\"')
        else:
            raise ValueError('FixBorder: \"dy\" must be the same length to \"ty\", or \"dy\" must be \"None\"')
    
    return clip

def FixBorderX(clip: VideoNode, target: int = 0, donor: Optional[int] = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    w = clip.width
    
    if target == donor or donor is None:
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
        clip = core.std.ShufflePlanes([(clip if i == plane else orig) for i in range(orig.format.num_planes)], [(0 if i == plane else i) for i in range(orig.format.num_planes)], orig.format.color_family)
    
    return clip

def FixBorderY(clip: VideoNode, target: int = 0, donor: Optional[int] = None, limit: int = 0, plane: int = 0) -> VideoNode:
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    h = clip.height
    
    if target == donor or donor is None:
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
        clip = core.std.ShufflePlanes([(clip if i == plane else orig) for i in range(orig.format.num_planes)], [(0 if i == plane else i) for i in range(orig.format.num_planes)], orig.format.color_family)
    
    return clip

def MaskDetail(clip: VideoNode, final_width: Optional[Union[float, int]] = None, final_height: Optional[Union[float, int]] = None, RGmode: int = 3,
               cutoff: int = 70, gain: float = 0.75, expandN: int = 2, inflateN: int = 1, blur_more: bool = False,
               src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None,
               kernel: str = 'bilinear', b: float = 0, c: float = 0.5, taps: int = 3, down: bool = True, frac: bool = True) -> VideoNode:
    
    '''
    MaskDetail by "Tada no Snob", ported from AviSynth version with minor additions.
    Has nothing to do with the port by MonoS.
    It is based on the rescale class from muvsfunc, therefore it supports fractional resolutions
    and automatic width calculation based on the original aspect ratio.
    "down = True" is added for backward compatibility and does not support fractional resolutions.
    Also, this option is incompatible with using odd resolutions when there is chroma subsampling in the source.
    '''
    
    if final_height is None:
        raise ValueError('MaskDetail: \"final_height\" must be specified')
    
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
    
    if final_width is None:
        if frac:
            resc = rescaler.rescale(clip, final_height, h)
        else:
            resc = rescaler.rescale(clip, final_height)
    else:
        if frac:
            desc = rescaler.descale(clip, final_width, final_height, h)
        else:
            desc = rescaler.descale(clip, final_width, final_height)
        resc = rescaler.upscale(desc, w, h)
    
    diff = core.std.MakeDiff(clip, resc)
    initial_mask = core.hist.Luma(diff).rgvs.RemoveGrain(RGmode)
    initial_mask = core.std.Expr(initial_mask, f'x {cutoff << step} < 0 x {gain} {full} x + {full} / * * ?')
    expanded = haf_mt_expand_multi(initial_mask, sw = expandN, sh = expandN)
    final = haf_mt_inflate_multi(expanded, radius = inflateN)
    
    if down:
        if final_width is None:
            raise ValueError('MaskDetail: if \"down\" is \"True\", then \"final_width\" can\'t be \"None\"')
        if not isinstance(final_width, int) or not isinstance(final_height, int):
            raise ValueError('MaskDetail: if \"down\" is \"True\", then \"final_width\" and \"final_height\" must be intgers')
        if space != GRAY and (final_width >> sub_w << sub_w != final_width or final_height >> sub_h << sub_h != final_height):
            raise ValueError('MaskDetail: \"final_width\" or \"final_height\" does not match the chroma subsampling of the output clip')
        if src_left is None:
            src_left = 0
        if src_top is None:
            src_top = 0
        if src_width is None:
            src_width = w
        elif src_width <= 0:
            src_width = w - src_left + src_width
        if src_height is None:
            src_height = h
        elif src_height <= 0:
            src_height = h - src_top + src_height
        final = core.resize.Bilinear(final, final_width, final_height, src_left = src_left, src_top = src_top, src_width = src_width, src_height = src_height)
    
    if blur_more:
        final = core.rgvs.RemoveGrain(final, 12)
    
    if space != GRAY:
        final = core.resize.Point(final, format = format_id)
    
    return final
