//! The byte-shuffle control both vector backends squeeze staged kinds with.

/// For each 8-bit "keep" mask, the byte indices of its set bits in ascending order, padded with
/// `0x80`. Used as a byte-shuffle control to squeeze 8 staged kinds down to the ones that survive:
/// `0x80` is out of range for both `vpshufb` and `TBL`, so the padding lanes come out zero either
/// way and the next group's store lands on top of them.
pub(super) static BCOMPACT: [[u8; 16]; 256] = bcompact();

const fn bcompact() -> [[u8; 16]; 256] {
    let mut t = [[0x80u8; 16]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let (mut b, mut k) = (0usize, 0usize);
        while b < 8 {
            if mask & (1 << b) != 0 {
                t[mask][k] = b as u8;
                k += 1;
            }
            b += 1;
        }
        mask += 1;
    }
    t
}
