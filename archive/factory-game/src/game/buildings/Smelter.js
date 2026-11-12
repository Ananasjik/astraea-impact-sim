class Smelter extends Building {
    constructor(x, y) {
        super(x, y, 'smelter');
        this.inventory = new Inventory(2); // slot 0 for inputs, 1 for outputs
        this.progress = 0;
        this.craftSpeed = 2; // seconds
        this.currentRecipe = null;
    }

    update(dt, world) {
        // Very simplified smelting: iron -> iron_plate, copper -> copper_wire
        const inputSlot = this.inventory.slots[0];
        
        if (inputSlot && inputSlot.count > 0) {
            if (!this.currentRecipe) {
                if (inputSlot.item === 'iron') this.currentRecipe = { input: 'iron', output: 'iron_plate' };
                else if (inputSlot.item === 'copper') this.currentRecipe = { input: 'copper', output: 'copper_wire' };
            }
        } else {
            this.currentRecipe = null;
            this.progress = 0;
        }

        if (this.currentRecipe) {
            this.progress += dt / 1000;
            if (this.progress >= this.craftSpeed) {
                // Try to add output
                if (this.inventory.addItem(this.currentRecipe.output, 1)) {
                    this.inventory.removeItem(this.currentRecipe.input, 1);
                    this.progress = 0;
                } else {
                    // Inventory full, stall
                    this.progress = this.craftSpeed;
                }
            }
        }
    }
}
