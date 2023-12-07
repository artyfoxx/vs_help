from vapoursynth import core, GRAY
from muvsfunc import Blur, haf_Clamp, haf_MinBlur, sbr, haf_mt_expand_multi, haf_mt_inflate_multi, haf_mt_deflate_multi

# Lanczos-based resize by "*.mp4 guy", ported from AviSynth version with minor additions and moved to fmtconv 
def autotap3(clip, dx = None, dy = None, mtaps3 = 1, thresh = 256):
    
    w = clip.width
    h = clip.height
    
    if dx is None:
        dx = w * 2
    
    if dy is None:
        dy = h * 2
    
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

# Dehalo by bion-x, ported from AviSynth version with minor additions 
def dehalo(clip, mode = 13, rep = True, rg = False, mask = 1, m = False):
    
    '''
    Supported formats: YUV 8 - 16 bit integer.
    mode = 1, 5, 11 - the weakest, artifacts will not cause.
    mode = 2, 3, 4 - bad modes, eat innocent parts, can not be used.
    mode = 10 - almost like mode = 1, 5, 11, but with a spread around the edges. I think it's a little better for noisy sources.
    mode = 14, 16, 17, 18 - the strongest of the "fit" ones, but they can blur the edges, mode = 13 is better.
    '''
    
    bits = clip.format.bits_per_sample
    mp0 = 1 << (bits - 6)
    mp1 = 1 << (bits - 8)
    mp2 = 1 << (bits - 1)
    
    e1 = core.std.Expr([core.std.Maximum(clip, planes = 0), core.std.Minimum(clip, planes = 0)], [f'x y - {mp0} - 4 *', ''])
    e2 = haf_mt_expand_multi(e1, planes = 0, sw = 2, sh = 2)
    e2 = core.std.Merge(e2, haf_mt_expand_multi(e2, planes = 0), [0.5, 0])
    e2 = haf_mt_inflate_multi(e2, planes = 0)
    e3 = core.std.Merge(e2, haf_mt_expand_multi(e2, planes = 0), [0.5, 0])
    e3 = core.std.Expr([e3, haf_mt_deflate_multi(e1, planes = 0)], ['x y 1.2 * -', ''])
    e3 = haf_mt_inflate_multi(e3, planes = 0)
    
    m0 = core.std.BoxBlur(clip, planes = 0, hradius = 2, vradius = 2)
    m0 = core.std.Expr([clip, m0], ['x y - abs 0 > x y - 0.3125 * x + x ?', ''])
    m1 = core.std.Expr([clip, m0], [f'x y - {mp1} - 128 *', ''])
    m1 = haf_mt_expand_multi(m1, planes = 0)
    m1 = haf_mt_inflate_multi(m1, planes = 0)
    m2 = haf_mt_expand_multi(m1, planes = 0, sw = 2, sh = 2)
    m3 = core.std.Expr([m1, m2], ['y x -', ''])
    m3 = core.rgvs.RemoveGrain(m3, [21, 0])
    m3 = haf_mt_expand_multi(m3, planes = 0)
    
    if mask == 1:
        pass
    elif mask == 2:
        e3 = m3
    elif mask == 3:
        e3 = core.std.Expr([e3, m3], ['x y min', ''])
    elif mask == 4:
        e3 = core.std.Expr([e3, m3], ['x y max', ''])
    else:
        raise ValueError('dehalo: Please use 1...4 mask type')
    
    blurr = haf_MinBlur(clip, 1)
    blurr = core.rgvs.RemoveGrain(blurr, [11, 0])
    blurr = core.rgvs.RemoveGrain(blurr, [11, 0])
    
    if rg:
        dh1 = core.rgvs.Repair(clip, core.rgvs.RemoveGrain(clip, [21]), [1])
        dh1 = core.std.MaskedMerge(dh1, blurr, e3, planes = 0)
    else:
        dh1 = core.std.MaskedMerge(clip, blurr, e3, planes = 0)
    
    dh1D = core.std.MakeDiff(clip, dh1, planes = 0)
    tmp = sbr(dh1)
    med2D = core.std.MakeDiff(tmp, core.ctmf.CTMF(tmp, 2, planes = 0), planes = 0)
    DD  = core.std.Expr([dh1D, med2D], [f'x {mp2} - y {mp2} - * 0 < {mp2} x {mp2} - abs y {mp2} - abs 2 * < x y {mp2} - 2 * {mp2} + ? ?', ''])
    dh2 = core.std.MergeDiff(dh1, DD, planes = 0)
    
    r = core.rgvs.Repair(clip, dh2, [mode]) if rep else dh2
    
    clip = haf_Clamp(clip, r, clip, 0, mp1 * 20, planes = 0)
    
    if m:
        clip = e3
    
    return clip

# Just a hardline znedi3 upscale with autotap3
def znedi3at(clip, target_width = None, target_height = None, src_left = None, src_top = None, src_width = None, src_height = None):
    
    if clip.format.color_family != GRAY:
        raise ValueError('znedi3at: Only GRAY clip is supported')
    
    if target_width is None:
        target_width = clip.width
    if target_height is None:
        target_height = clip.height
    if src_left is None:
        src_left = 0
    if src_top is None:
        src_top = 0
    if src_width is None:
        src_width = clip.width
    elif src_width <= 0:
        src_width = clip.width - src_left + src_width
    if src_height is None:
        src_height = clip.height
    elif src_height <= 0:
        src_height = clip.height - src_top + src_height
    
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
def FixBorderX(clip, target = 0, donor = 0, limit = 0, plane = 0):
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    w = clip.width
    
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

def FixBorderY(clip, target = 0, donor = 0, limit = 0, plane = 0):
    
    space = clip.format.color_family
    if space != GRAY:
        orig = clip
        clip = core.std.ShufflePlanes(clip, plane, GRAY)
    
    h = clip.height
    
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
