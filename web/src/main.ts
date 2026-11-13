import { SimulationApp } from './ui/SimulationApp';

function bootstrap(): SimulationApp | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const canvasContainer =
    document.getElementById('canvas-container') ?? document.body;
  const uiContainer =
    document.getElementById('ui-container') ?? document.body;

  if (canvasContainer === null || uiContainer === null) {
    return null;
  }

  const app = new SimulationApp(canvasContainer, uiContainer);
  app.start();

  const onUnload = (): void => {
    app.dispose();
  };
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', onUnload, { once: true });
  }

  const hot = (
    import.meta as unknown as {
      hot?: { dispose: (cb: () => void) => void };
    }
  ).hot;
  if (hot !== undefined && typeof hot.dispose === 'function') {
    hot.dispose(() => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('beforeunload', onUnload);
      }
      app.dispose();
    });
  }

  return app;
}

bootstrap();
