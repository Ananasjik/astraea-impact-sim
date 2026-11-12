class Inventory {
    constructor(size) {
        this.size = size;
        this.slots = new Array(size).fill(null); // { item: 'id', count: 1 }
    }

    addItem(itemId, count = 1) {
        // Find existing stack
        for (let i = 0; i < this.size; i++) {
            if (this.slots[i] && this.slots[i].item === itemId) {
                // Stack limit logic can go here (e.g. 64)
                this.slots[i].count += count;
                return true;
            }
        }
        
        // Find empty slot
        for (let i = 0; i < this.size; i++) {
            if (!this.slots[i]) {
                this.slots[i] = { item: itemId, count: count };
                return true;
            }
        }
        
        return false; // Inventory full
    }

    removeItem(itemId, count = 1) {
        let removed = 0;
        for (let i = 0; i < this.size; i++) {
            if (this.slots[i] && this.slots[i].item === itemId) {
                const toTake = Math.min(count - removed, this.slots[i].count);
                this.slots[i].count -= toTake;
                removed += toTake;
                
                if (this.slots[i].count <= 0) {
                    this.slots[i] = null;
                }
                
                if (removed >= count) return true;
            }
        }
        return removed === count;
    }

    hasItem(itemId, count = 1) {
        let found = 0;
        for (let i = 0; i < this.size; i++) {
            if (this.slots[i] && this.slots[i].item === itemId) {
                found += this.slots[i].count;
                if (found >= count) return true;
            }
        }
        return false;
    }
}
