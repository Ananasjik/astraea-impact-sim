class Conveyor extends Building {
    constructor(x, y, direction = 0) { // 0: Right, 1: Down, 2: Left, 3: Up
        super(x, y, 'conveyor');
        this.direction = direction;
        this.items = []; // { itemId, position: 0 to 1 }
        this.speed = 1.5; // tiles per second
    }

    update(dt, world) {
        const dtSec = dt / 1000;
        
        // Move items
        for (let i = this.items.length - 1; i >= 0; i--) {
            const item = this.items[i];
            
            // If item is at the end of the belt
            if (item.position >= 1) {
                if (this.tryOutputItem(item, world)) {
                    this.items.splice(i, 1);
                }
            } else {
                // Check if blocked by item ahead
                const nextItem = i > 0 ? this.items[i - 1] : null;
                const limit = nextItem ? nextItem.position - 0.2 : 1;
                
                if (item.position < limit) {
                    item.position = Math.min(limit, item.position + this.speed * dtSec);
                }
            }
        }

        // Pull from miner if we have space at the start
        if (this.items.length === 0 || this.items[this.items.length - 1].position > 0.2) {
            // Check behind based on direction
            let behindX = this.x;
            let behindY = this.y;
            if (this.direction === 0) behindX -= 1;
            else if (this.direction === 1) behindY -= 1;
            else if (this.direction === 2) behindX += 1;
            else if (this.direction === 3) behindY += 1;

            const bTile = world.getTile(behindX, behindY);
            if (bTile && bTile.building) {
                const b = bTile.building;
                if (b.type === 'miner' && b.inventory.slots[0]) {
                    this.items.push({ itemId: b.inventory.slots[0].item, position: 0 });
                    b.inventory.removeItem(b.inventory.slots[0].item, 1);
                }
            }
        }
    }

    tryOutputItem(item, world) {
        let nextX = this.x;
        let nextY = this.y;
        if (this.direction === 0) nextX += 1;
        else if (this.direction === 1) nextY += 1;
        else if (this.direction === 2) nextX -= 1;
        else if (this.direction === 3) nextY -= 1;

        const nTile = world.getTile(nextX, nextY);
        if (!nTile || !nTile.building) return false;
        
        const b = nTile.building;
        
        if (b.type === 'conveyor') {
            // Check if there is space on the next conveyor
            if (b.items.length === 0 || b.items[b.items.length - 1].position > 0.2) {
                b.items.push({ itemId: item.itemId, position: 0 });
                return true;
            }
        } else if (b.type === 'smelter') {
            return b.inventory.addItem(item.itemId, 1);
        }
        
        return false;
    }
}
