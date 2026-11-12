// A simple PRNG and Noise function for world generation
class Utils {
    // Very simple pseudo-random number generator
    static mulberry32(a) {
        return function() {
          var t = a += 0x6D2B79F5;
          t = Math.imul(t ^ t >>> 15, t | 1);
          t ^= t + Math.imul(t ^ t >>> 7, t | 61);
          return ((t ^ t >>> 14) >>> 0) / 4294967296;
        }
    }

    // Simple 2D Value Noise (Not as good as Perlin/Simplex, but tiny and works without external libs)
    static noise2D(x, y, seed = 1) {
        const dot = (x, y, vx, vy) => x * vx + y * vy;
        
        // Pseudo-random gradient vector
        const grad = (ix, iy) => {
            const random = Utils.mulberry32(ix * 1337 + iy * 420 + seed)();
            const angle = random * Math.PI * 2;
            return { x: Math.cos(angle), y: Math.sin(angle) };
        };

        const ix = Math.floor(x);
        const iy = Math.floor(y);
        const fx = x - ix;
        const fy = y - iy;

        const g00 = grad(ix, iy);
        const g10 = grad(ix + 1, iy);
        const g01 = grad(ix, iy + 1);
        const g11 = grad(ix + 1, iy + 1);

        const d00 = dot(fx, fy, g00.x, g00.y);
        const d10 = dot(fx - 1, fy, g10.x, g10.y);
        const d01 = dot(fx, fy - 1, g01.x, g01.y);
        const d11 = dot(fx - 1, fy - 1, g11.x, g11.y);

        const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
        const lerp = (a, b, t) => a + t * (b - a);

        const u = fade(fx);
        const v = fade(fy);

        const x1 = lerp(d00, d10, u);
        const x2 = lerp(d01, d11, u);
        return lerp(x1, x2, v); // Returns roughly -1 to 1
    }
}
