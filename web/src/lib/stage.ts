import * as THREE from "three";

/**
 * The WebGL boilerplate, once.
 *
 * Both 3D scenes need the same five things — a renderer sized to its container, a camera
 * that recomposes at narrow widths, a loop that stops when the section scrolls away or the
 * tab hides, pointer parallax, and a dispose that actually releases the context. Written
 * twice, those drift; the second scene forgets to stop its loop and the page burns a core
 * on a picture nobody is looking at.
 *
 * This laptop runs a 7B model on CPU. Frames are not free.
 */
export interface Stage {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  /** Everything you add should go in here — it carries the pointer parallax. */
  world: THREE.Group;
  /** Smoothed pointer, -1..1 on each axis. */
  pointer: { x: number; y: number };
  /** Register the per-frame callback. Receives seconds elapsed and delta. */
  onFrame: (fn: (t: number, dt: number) => void) => void;
  /** Called on resize with the container's pixel width. Set camera distance here. */
  onResize: (fn: (w: number, h: number) => void) => void;
  dispose: () => void;
}

export function createStage(
  el: HTMLElement,
  opts: {
    reduced: boolean;
    fov?: number;
    /** How far the pointer moves the world, in radians. 0 disables parallax. */
    parallax?: number;
    /** Frozen time used for the single frame rendered under reduced motion. */
    staticTime?: number;
  },
): Stage | null {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      powerPreference: "low-power",
    });
  } catch {
    // No WebGL. Every scene on this site sits behind a CSS gradient that reads fine
    // without it, so this is a degraded page, not a broken one.
    return null;
  }

  const dpr = Math.min(window.devicePixelRatio, 1.75);
  renderer.setPixelRatio(dpr);
  renderer.setClearColor(0x000000, 0);
  el.appendChild(renderer.domElement);
  renderer.domElement.style.cssText = "width:100%;height:100%;display:block";

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(opts.fov ?? 42, 1, 0.1, 120);
  const world = new THREE.Group();
  scene.add(world);

  const pointer = { x: 0, y: 0 };
  const target = { x: 0, y: 0 };
  const parallax = opts.parallax ?? 0.14;

  const frameFns: ((t: number, dt: number) => void)[] = [];
  const resizeFns: ((w: number, h: number) => void)[] = [];

  const resize = () => {
    const w = el.clientWidth || 1;
    const h = el.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    for (const fn of resizeFns) fn(w, h);
    camera.updateProjectionMatrix();
  };

  const ro = new ResizeObserver(resize);
  ro.observe(el);

  const onPointer = (ev: PointerEvent) => {
    const r = el.getBoundingClientRect();
    target.x = ((ev.clientX - r.left) / r.width - 0.5) * 2;
    target.y = ((ev.clientY - r.top) / r.height - 0.5) * 2;
  };
  if (!opts.reduced && parallax > 0) {
    window.addEventListener("pointermove", onPointer, { passive: true });
  }

  const clock = new THREE.Clock(false);
  let raf = 0;
  let running = false;
  let elapsed = opts.reduced ? (opts.staticTime ?? 6.2) : 0;

  const render = () => {
    renderer.render(scene, camera);
  };

  const frame = () => {
    const dt = Math.min(clock.getDelta(), 0.05);
    elapsed += dt;
    pointer.x += (target.x - pointer.x) * 0.05;
    pointer.y += (target.y - pointer.y) * 0.05;
    if (parallax > 0) {
      world.rotation.y += (pointer.x * parallax - world.rotation.y) * 0.06;
      world.rotation.x += (-pointer.y * parallax * 0.7 - world.rotation.x) * 0.06;
    }
    for (const fn of frameFns) fn(elapsed, dt);
    render();
    raf = requestAnimationFrame(frame);
  };

  const start = () => {
    if (running || opts.reduced) return;
    running = true;
    clock.start();
    raf = requestAnimationFrame(frame);
  };
  const stop = () => {
    running = false;
    clock.stop();
    cancelAnimationFrame(raf);
  };

  const io = new IntersectionObserver(([e]) => (e.isIntersecting ? start() : stop()), {
    threshold: 0.02,
  });
  io.observe(el);

  const onVisibility = () => (document.hidden ? stop() : start());
  document.addEventListener("visibilitychange", onVisibility);

  resize();

  return {
    scene,
    camera,
    renderer,
    world,
    pointer,
    onFrame: (fn) => {
      frameFns.push(fn);
    },
    onResize: (fn) => {
      resizeFns.push(fn);
      fn(el.clientWidth || 1, el.clientHeight || 1);
      camera.updateProjectionMatrix();
    },
    dispose: () => {
      stop();
      io.disconnect();
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pointermove", onPointer);
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat?.dispose();
      });
      renderer.dispose();
      if (renderer.domElement.parentElement === el) el.removeChild(renderer.domElement);
    },
  };
}

/**
 * Deterministic RNG.
 *
 * Every other layer of this project regenerates byte-identically from a seed; a hero that
 * scatters its geometry differently on every reload would be the one part you could not
 * take a reference screenshot of.
 */
export function seededRandom(seed = 0x2f6e2b1): () => number {
  let s = seed;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 100000) / 100000;
  };
}

/** Renders one frame immediately — used so a reduced-motion visitor still sees the scene. */
export function paintOnce(stage: Stage) {
  stage.renderer.render(stage.scene, stage.camera);
}
