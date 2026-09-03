import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useReducedMotion } from "../lib/hooks";
import { createStage, paintOnce, seededRandom } from "../lib/stage";

/**
 * The pipeline as a cyclone.
 *
 * Money enters at the wide mouth and spirals down through a stack of narrowing rings — one
 * ring per stage of the pipeline. Each particle is assigned the ring it stops at from the
 * **real stage counts**, so the attrition you watch is the attrition that was measured: if
 * 51 of 308 deductions are abstained to a human, one particle in six parks at the
 * classification ring and goes cold.
 *
 * The spin is not decoration either. Angular velocity rises as the radius shrinks — the
 * same reason a real vortex accelerates toward its throat — so the motion tightens exactly
 * where the funnel does. It makes the narrowing legible without a single label.
 *
 * A bar chart carries the same numbers. This carries what the bar chart cannot: that most
 * of what goes in never reaches the bottom, and that this is the system working.
 */

export interface SieveStage {
  label: string;
  count: number;
}

const VERT = /* glsl */ `
  attribute float aStart;
  attribute float aAngle;
  attribute float aSpeed;
  attribute float aStop;    // descent progress at which this particle is filtered out
  attribute float aSize;
  attribute float aWobble;

  uniform float uTime;
  uniform float uLag;       // seconds behind the head of the trail
  uniform float uAlpha;     // trail segments render dimmer than the head
  uniform float uHeight;
  uniform float uTop;
  uniform float uRTop;
  uniform float uRBot;
  uniform float uPR;
  uniform float uSize;
  uniform vec3  uGold;
  uniform vec3  uJade;
  uniform vec3  uMuted;

  varying vec3  vColor;
  varying float vAlpha;

  void main() {
    float t = uTime - uLag;
    float prog = fract(aStart + t * aSpeed);
    float parked = step(aStop, prog);
    float p = min(prog, aStop);

    // A cyclone, not a carousel. The 1/r term is conservation of angular momentum: the
    // spin accelerates into the throat, which is what makes the narrowing readable.
    float r = mix(uRTop, uRBot, p);
    float ang = aAngle + t * (0.5 + 0.85 / max(r, 0.35)) + p * 1.6;

    // A little radial breathing so the wall is a moving sheet rather than a wire cylinder.
    float wob = sin(t * 1.7 + aWobble * 6.2831) * 0.055 * r;

    float y = uTop - p * uHeight + sin(t * 2.1 + aWobble * 6.2831) * 0.03;
    vec3 world = vec3(cos(ang) * (r + wob), y, sin(ang) * (r + wob));

    // Fade out just after parking, then recycle at the top. Without this every ring grows a
    // bright collar of stalled particles and the shape stops reading.
    float after = clamp((prog - aStop) / 0.14, 0.0, 1.0);
    float alpha = (1.0 - after) * uAlpha;
    alpha *= smoothstep(0.0, 0.05, prog);

    float survives = step(0.995, aStop);
    vec3 col = mix(uGold, uMuted, parked * 0.75);
    col = mix(col, uJade, survives * smoothstep(0.72, 1.0, p));

    vec4 mv = modelViewMatrix * vec4(world, 1.0);
    gl_Position  = projectionMatrix * mv;
    gl_PointSize = uSize * aSize * uPR * (1.0 / -mv.z) * 9.0;

    vColor = col;
    vAlpha = alpha;
  }
`;

const FRAG = /* glsl */ `
  precision mediump float;
  varying vec3  vColor;
  varying float vAlpha;
  void main() {
    vec2  d = gl_PointCoord - 0.5;
    float r = length(d);
    if (r > 0.5) discard;
    gl_FragColor = vec4(vColor, smoothstep(0.5, 0.06, r) * vAlpha);
  }
`;

function ringGeometry(radius: number, segments = 128): THREE.BufferGeometry {
  const pts: number[] = [];
  for (let i = 0; i < segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const b = ((i + 1) / segments) * Math.PI * 2;
    pts.push(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    pts.push(Math.cos(b) * radius, 0, Math.sin(b) * radius);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return g;
}

/** A dashed ring: the same circle with alternate segments dropped. Reads as a gauge. */
function tickRingGeometry(radius: number, ticks = 48, len = 0.5): THREE.BufferGeometry {
  const pts: number[] = [];
  for (let i = 0; i < ticks; i++) {
    const a = (i / ticks) * Math.PI * 2;
    const b = a + ((Math.PI * 2) / ticks) * len;
    pts.push(Math.cos(a) * radius, 0, Math.sin(a) * radius);
    pts.push(Math.cos(b) * radius, 0, Math.sin(b) * radius);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return g;
}

export default function SieveScene({ stages }: { stages: SieveStage[] }) {
  const host = useRef<HTMLDivElement | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = host.current;
    if (!el || stages.length < 2) return;

    const stage = createStage(el, { reduced, fov: 38, parallax: 0.24 });
    if (!stage) return;

    const { scene, camera, world } = stage;
    scene.fog = new THREE.FogExp2(0x05060a, 0.026);
    camera.position.set(0, 2.1, 11.5);
    camera.lookAt(0, -0.6, 0);

    const rnd = seededRandom(0x51ee7e);
    const GOLD = 0xe8b24c;
    const JADE = 0x5ad9a2;

    const TOP = 4.2;
    const HEIGHT = 8.6;
    const R_TOP = 3.5;
    const R_BOT = 0.62;
    const N = stages.length;
    const radiusAt = (p: number) => R_TOP + (R_BOT - R_TOP) * p;

    const disposables: { dispose(): void }[] = [];
    const ringGroup = new THREE.Group();
    world.add(ringGroup);

    /* ---- minor rings: the wall of the cyclone ---- */
    // Between every pair of stages, so the cone has a surface rather than nine hoops in
    // empty space. Faint enough to read as structure, not as more data.
    for (let i = 0; i < (N - 1) * 4; i++) {
      const p = i / ((N - 1) * 4);
      const geo = ringGeometry(radiusAt(p), 72);
      disposables.push(geo);
      const line = new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: GOLD,
          transparent: true,
          opacity: 0.035,
        }),
      );
      line.position.y = TOP - p * HEIGHT;
      ringGroup.add(line);
    }

    /* ---- stage rings ---- */
    const stageRings: {
      line: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
      ticks: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
      base: number;
      y: number;
    }[] = [];

    for (let i = 0; i < N; i++) {
      const p = i / (N - 1);
      const r = radiusAt(p);
      const y = TOP - p * HEIGHT;
      const last = i === N - 1;

      const geo = ringGeometry(r);
      const tickGeo = tickRingGeometry(r * 1.045, 40, 0.34);
      disposables.push(geo, tickGeo);

      const line = new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({
          color: last ? JADE : GOLD,
          transparent: true,
          opacity: last ? 0.62 : 0.2 + (1 - p) * 0.16,
        }),
      );
      line.position.y = y;
      ringGroup.add(line);

      // Gauge ticks outside each stage ring, counter-rotating. They give the cone a sense
      // of scale and make the rotation visible even where particles are sparse.
      const ticks = new THREE.LineSegments(
        tickGeo,
        new THREE.LineBasicMaterial({
          color: last ? JADE : GOLD,
          transparent: true,
          opacity: 0.1,
        }),
      );
      ticks.position.y = y;
      ringGroup.add(ticks);

      stageRings.push({ line, ticks, base: line.material.opacity, y });

      const discGeo = new THREE.CircleGeometry(r, 64);
      disposables.push(discGeo);
      const disc = new THREE.Mesh(
        discGeo,
        new THREE.MeshBasicMaterial({
          color: last ? JADE : GOLD,
          transparent: true,
          opacity: last ? 0.055 : 0.014,
          depthWrite: false,
          side: THREE.DoubleSide,
          blending: THREE.AdditiveBlending,
        }),
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.y = y;
      ringGroup.add(disc);
    }

    /* ---- the throat: a glowing column down the axis ---- */
    // `BackSide` only, which is the trick that turns a cylinder into a glow rather than a
    // pillar: the near wall is culled, so what you see is the far wall through the tube,
    // brightest where the surface is edge-on. Rendering both sides additively drew the
    // column twice and put a solid gold bar down the middle of the cyclone.
    const coreGeo = new THREE.CylinderGeometry(0.05, 0.13, HEIGHT * 0.96, 16, 1, true);
    disposables.push(coreGeo);
    const core = new THREE.Mesh(
      coreGeo,
      new THREE.MeshBasicMaterial({
        color: GOLD,
        transparent: true,
        opacity: 0.055,
        depthWrite: false,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      }),
    );
    core.position.y = TOP - HEIGHT / 2;
    world.add(core);

    /* ---- silhouette spokes ---- */
    const spokes: number[] = [];
    for (let i = 0; i < 32; i++) {
      const a = (i / 32) * Math.PI * 2;
      // Curved, not straight: a spiral edge sells rotation even when the scene is still.
      for (let s = 0; s < 12; s++) {
        const p0 = s / 12;
        const p1 = (s + 1) / 12;
        const a0 = a + p0 * 1.6;
        const a1 = a + p1 * 1.6;
        spokes.push(
          Math.cos(a0) * radiusAt(p0), TOP - p0 * HEIGHT, Math.sin(a0) * radiusAt(p0),
          Math.cos(a1) * radiusAt(p1), TOP - p1 * HEIGHT, Math.sin(a1) * radiusAt(p1),
        );
      }
    }
    const spokeGeo = new THREE.BufferGeometry();
    spokeGeo.setAttribute("position", new THREE.Float32BufferAttribute(spokes, 3));
    disposables.push(spokeGeo);
    const spokeLines = new THREE.LineSegments(
      spokeGeo,
      new THREE.LineBasicMaterial({ color: GOLD, transparent: true, opacity: 0.085 }),
    );
    ringGroup.add(spokeLines);

    /* ---- particles, distributed by the measured attrition ---- */
    const intake = Math.max(1, stages[0].count);
    const survival = stages.map((s) => Math.max(0, Math.min(1, s.count / intake)));

    const mobile = window.matchMedia("(max-width: 767px)").matches;
    const COUNT = mobile ? 2400 : 5200;

    const pos = new Float32Array(COUNT * 3); // required by three, unused by the shader
    const start = new Float32Array(COUNT);
    const angle = new Float32Array(COUNT);
    const speed = new Float32Array(COUNT);
    const stop = new Float32Array(COUNT);
    const size = new Float32Array(COUNT);
    const wobble = new Float32Array(COUNT);

    for (let i = 0; i < COUNT; i++) {
      const roll = rnd();
      let stopAt = 1.0;
      for (let s = 1; s < N; s++) {
        if (roll > survival[s]) {
          stopAt = (s - 1) / (N - 1) + 0.02;
          break;
        }
      }
      stop[i] = Math.min(stopAt, 1.0);
      start[i] = rnd();
      angle[i] = rnd() * Math.PI * 2;
      speed[i] = 0.07 + rnd() * 0.055;
      size[i] = 0.5 + rnd() * 0.8;
      wobble[i] = rnd();
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aStart", new THREE.BufferAttribute(start, 1));
    geo.setAttribute("aAngle", new THREE.BufferAttribute(angle, 1));
    geo.setAttribute("aSpeed", new THREE.BufferAttribute(speed, 1));
    geo.setAttribute("aStop", new THREE.BufferAttribute(stop, 1));
    geo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
    geo.setAttribute("aWobble", new THREE.BufferAttribute(wobble, 1));
    disposables.push(geo);

    const shared = {
      uTime: { value: reduced ? 8.0 : 0 },
      uHeight: { value: HEIGHT },
      uTop: { value: TOP },
      uRTop: { value: R_TOP },
      uRBot: { value: R_BOT },
      uPR: { value: Math.min(window.devicePixelRatio, 1.75) },
      uSize: { value: mobile ? 3.0 : 3.8 },
      uGold: { value: new THREE.Color("#e8b24c") },
      uJade: { value: new THREE.Color("#5ad9a2") },
      // Filtered-out money goes cold rather than red: most of what this funnel removes was
      // never ours to chase, and colouring it as loss would be the wrong claim.
      uMuted: { value: new THREE.Color("#414a5c") },
    };

    // Motion trails, done the cheap way: the same geometry drawn three more times at small
    // time offsets. `Points` cannot draw a streak, and a real trail buffer would mean
    // storing history per particle — four draw calls of one buffer costs nothing and reads
    // as speed, which is the whole point of a cyclone.
    const TRAIL = [
      { lag: 0, alpha: 1 },
      { lag: 0.05, alpha: 0.42 },
      { lag: 0.1, alpha: 0.2 },
      { lag: 0.16, alpha: 0.09 },
    ];
    const layers = TRAIL.map(({ lag, alpha }) => {
      const uniforms = {
        ...shared,
        uLag: { value: lag },
        uAlpha: { value: alpha },
      };
      const mat = new THREE.ShaderMaterial({
        uniforms,
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending,
      });
      const points = new THREE.Points(geo, mat);
      points.frustumCulled = false;
      world.add(points);
      return { points, mat };
    });

    /* ---- the pool that collects at the mouth ---- */
    const poolY = TOP - HEIGHT - 0.05;
    const poolGeo = new THREE.CircleGeometry(1.5, 48);
    disposables.push(poolGeo);
    const pool = new THREE.Mesh(
      poolGeo,
      new THREE.MeshBasicMaterial({
        color: JADE,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    pool.rotation.x = -Math.PI / 2;
    pool.position.y = poolY;
    world.add(pool);

    // Ripples leaving the pool — money landing, three times a cycle.
    const ripples = [0, 1, 2].map((i) => {
      const g = ringGeometry(1, 64);
      disposables.push(g);
      const m = new THREE.LineSegments(
        g,
        new THREE.LineBasicMaterial({ color: JADE, transparent: true, opacity: 0.3 }),
      );
      m.position.y = poolY;
      world.add(m);
      return { mesh: m, phase: i / 3 };
    });

    stage.onResize((w) => {
      camera.position.z = w < 560 ? 15.5 : w < 900 ? 13.2 : 11.5;
      camera.lookAt(0, -0.6, 0);
    });

    stage.onFrame((t) => {
      shared.uTime.value = t;

      // The whole cone turns slowly against the particles' own spin, so the structure and
      // its contents never lock into a single rigid rotation.
      ringGroup.rotation.y = t * 0.05;
      spokeLines.rotation.y = -t * 0.02;
      core.rotation.y = t * 0.4;
      core.material.opacity = 0.05 + Math.sin(t * 1.6) * 0.018;

      for (let i = 0; i < stageRings.length; i++) {
        const r = stageRings[i];
        // A pulse travelling down the stack — the cursor running through the pipeline.
        const wave = Math.sin(t * 0.9 - i * 0.62);
        r.line.material.opacity = r.base * (0.72 + 0.38 * wave);
        r.ticks.rotation.y = -t * (0.1 + i * 0.03);
        r.ticks.material.opacity = 0.07 + 0.06 * Math.max(0, wave);
      }

      for (const rp of ripples) {
        const phase = (t * 0.34 + rp.phase) % 1;
        const s = 0.5 + phase * 2.6;
        rp.mesh.scale.setScalar(s);
        rp.mesh.material.opacity = 0.3 * (1 - phase) * (1 - phase);
      }

      pool.material.opacity = 0.07 + Math.sin(t * 1.4) * 0.022;
    });

    paintOnce(stage);

    return () => {
      for (const d of disposables) d.dispose();
      for (const l of layers) l.mat.dispose();
      stage.dispose();
    };
  }, [reduced, stages]);

  return <div ref={host} className="absolute inset-0" aria-hidden="true" />;
}
