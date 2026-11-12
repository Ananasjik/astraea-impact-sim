const CHUNK_SIZE = 16;

class World {
    constructor() {
        this.chunks = {}; // map of "x,y" to chunk data
        this.seed = Math.random() * 10000;
    }

    getChunkKey(cx, cy) {
        return `${cx},${cy}`;
    }

    getTile(tx, ty) {
        const cx = Math.floor(tx / CHUNK_SIZE);
        const cy = Math.floor(ty / CHUNK_SIZE);
        const key = this.getChunkKey(cx, cy);

        if (!this.chunks[key]) {
            this.generateChunk(cx, cy);
        }

        const chunk = this.chunks[key];
        const localX = tx - cx * CHUNK_SIZE;
        const localY = ty - cy * CHUNK_SIZE;
        return chunk.tiles[localX][localY];
    }

    setBuilding(tx, ty, building) {
        const tile = this.getTile(tx, ty);
        tile.building = building;
    }
    
    getBuilding(tx, ty) {
        return this.getTile(tx, ty).building;
    }

    canPlaceBuilding(tx, ty) {
        const tile = this.getTile(tx, ty);
        return !tile.building && tile.type !== 'water';
    }

    generateChunk(cx, cy) {
        const chunk = {
            x: cx,
            y: cy,
            tiles: []
        };

        for (let x = 0; x < CHUNK_SIZE; x++) {
            chunk.tiles[x] = [];
            for (let y = 0; y < CHUNK_SIZE; y++) {
                const worldX = cx * CHUNK_SIZE + x;
                const worldY = cy * CHUNK_SIZE + y;
                
                // Terrain noise
                const scale = 0.05;
                const n = Utils.noise2D(worldX * scale, worldY * scale, this.seed);
                
                let type = 'grass';
                if (n < -0.3) type = 'water';
                else if (n > 0.4) type = 'stone';
                
                // Ore noise
                let ore = null;
                if (type === 'grass' || type === 'stone') {
                    const oreScale = 0.1;
                    const coalNoise = Utils.noise2D(worldX * oreScale, worldY * oreScale, this.seed + 100);
                    const ironNoise = Utils.noise2D(worldX * oreScale, worldY * oreScale, this.seed + 200);
                    const copperNoise = Utils.noise2D(worldX * oreScale, worldY * oreScale, this.seed + 300);
                    
                    if (coalNoise > 0.6) ore = 'coal';
                    else if (ironNoise > 0.6) ore = 'iron';
                    else if (copperNoise > 0.6) ore = 'copper';
                }

                chunk.tiles[x][y] = {
                    type,
                    ore,
                    building: null
                };
            }
        }

        this.chunks[this.getChunkKey(cx, cy)] = chunk;
    }

    update(dt, player) {
        // Dynamic loading could happen here, but for now we just load when rendering or requesting
    }
}
