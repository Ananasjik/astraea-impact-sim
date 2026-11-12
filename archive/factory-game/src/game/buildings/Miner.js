class Miner extends Building {
    constructor(x, y) {
        super(x, y, 'miner');
        this.progress = 0;
        this.speed = 1; // items per second
        this.inventory = new Inventory(1);
    }

    update(dt, world) {
        if (this.inventory.slots[0]) {
            // Inventory full, can't mine more until emptied (by a belt)
            return;
        }

        const tile = world.getTile(this.x, this.y);
        if (tile.ore) {
            this.progress += dt / 1000;
            if (this.progress >= 1 / this.speed) {
                this.progress = 0;
                this.inventory.addItem(tile.ore, 1);
            }
        }
    }
}
