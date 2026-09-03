import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useReducedMotion } from "../lib/hooks";
import { createStage, paintOnce, seededRandom } from "../lib/stage";

/**
 * The hero: an invoice hall.
 *
 * A ruled floor running back into fog, invoices hanging in the dark above it, and a stream
 * of money crossing the room. At a gate in the middle the stream splits three ways, in the
 * proportions the run actually measured:
 *
 *   rising, in slate    money that was legitimately the buyer's — statutory withholding,
 *                       an agreed rebate, a credit note that really exists
 *   level, in jade      money the agent got back, or handed to a person who could
 *   falling, in crimson money still on the table
 *
 * The shares are passed in from the snapshot. If the run changes, the picture changes;
 * there is no number in this file.
 */

export interface LeakShares {
  valid: number;
  addressed: number;
  lost: number;
}

/* ---------------------------------------------------------------------------- floor -- */
const FLOOR_VERT = /* glsl */ `
  varying vec2 vWorld;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorld = wp.xz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const FLOOR_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vWorld;
  uniform vec3  uLine;
  uniform float uTime;

  // Screen-space derivative keeps the ruling one pixel wide at every depth, so the floor
  // reads as a ledger rather than dissolving into moire at the horizon.
  float ruling(vec2 p, float spacing) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) / (fwidth(p / spacing) + 1e-5);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    float fine  = ruling(vWorld, 1.0);
    float major = ruling(vWorld, 6.0);

    float dist = length(vWorld - vec2(0.0, 6.0));
    float fade = smoothstep(46.0, 6.0, dist);

    // A slow pulse travelling away from the viewer, like a cursor running down a column.
    float sweep = smoothstep(0.90, 1.0, sin(vWorld.y * 0.10 - uTime * 0.22) * 0.5 + 0.5);

    float a = (fine * 0.26 + major * 0.52 + sweep * fine * 0.7) * fade;
    if (a < 0.004) discard;
    gl_FragColor = vec4(uLine, a);
  }
`;

/* ---------------------------------------------------------------------------- sheets -- */
const SHEET_FRAG = /* glsl */ `
  precision mediump float;
  varying vec2 vUv;
  uniform vec3  uAccent;
  uniform float uOpacity;
  uniform float uSeed;

  float hash(float n) { return fract(sin(n * 12.9898 + uSeed) * 43758.5453); }

  void main() {
    vec2 uv = vUv;

    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float border = smoothstep(0.016, 0.002, edge);

    // Ruled rows of varying length: an invoice at a distance, without a texture file.
    float rows = 11.0;
    float y = (1.0 - uv.y) * rows;
    float rowId = floor(y);
    float band = smoothstep(0.60, 0.34, abs(fract(y) - 0.5) * 2.0);
    float len = 0.24 + 0.46 * hash(rowId);
    float body = band * step(0.10, uv.x) * step(uv.x, 0.10 + len);

    // One right-aligned figure per sheet — the amount, which is what the whole page is about.
    float amountRow = step(2.0, rowId) * step(rowId, 2.99);
    float amount = band * amountRow * step(0.62, uv.x) * step(uv.x, 0.90);

    // A rule under the header.
    float head = smoothstep(0.03, 0.0, abs(uv.y - 0.86));

    vec3 col = mix(vec3(0.62, 0.64, 0.70), vec3(1.0), body * 0.6);
    col = mix(col, uAccent, max(max(border, head * 0.7), amount));

    float a = uOpacity * (0.055 + body * 0.30 + border * 0.55 + amount * 0.65 + head * 0.35);
    if (a < 0.004) discard;
    gl_FragColor = vec4(col, a);
  }
`;

const SHEET_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/* --------------------------------------------------------------------------- stream -- */
const STREAM_VERT = /* glsl */ `
  attribute float aKind;
  attribute float aSpeed;
  attribute float aSize;

  uniform float uTime;
  uniform float uSpan;
  uniform float uGate;
  uniform float uSize;
  uniform float uPR;
  uniform vec3  uGold;
  uniform vec3  uSlate;
  uniform vec3  uJade;
  uniform vec3  uLeak;

  varying vec3  vColor;
  varying float vAlpha;

  void main() {
    // position carries (seed, lane, depth) so a particle needs no attribute beyond the
    // three three.js already allocates.
    float seed  = position.x;
    float lane  = position.y;
    float depth = position.z;

    float x  = mod(seed * uSpan + uTime * aSpeed, uSpan) - uSpan * 0.5;
    float y0 = lane * 0.30 + sin(x * 0.7 + seed * 40.0) * 0.05;

    // A short ramp, then three level rails. A long ramp reads as a spray leaving the
    // frame; the fork only reads as a fork if each branch settles and runs.
    float t = clamp((x - uGate) / 1.5, 0.0, 1.0);
    float e = t * t * (3.0 - 2.0 * t);

    float y     = y0;
    vec3  col   = uGold;
    float alpha = 1.0;

    if (aKind < 0.5) {
      y   = mix(y0, 1.42 + lane * 0.045, e);
      col = mix(uGold, uSlate, min(1.0, e * 1.5));
    } else if (aKind < 1.5) {
      y   = mix(y0, -0.46 + lane * 0.04, e);
      col = mix(uGold, uJade, min(1.0, e * 1.5));
    } else {
      y     = y0 - 2.6 * e * e;
      col   = mix(uGold, uLeak, min(1.0, e * 1.8));
      alpha = 1.0 - smoothstep(0.45, 1.0, t);
    }

    alpha *= smoothstep(0.0, 0.07, (x + uSpan * 0.5) / uSpan);

    vec4 mv = modelViewMatrix * vec4(x, y, depth * 0.55, 1.0);
    gl_Position  = projectionMatrix * mv;
    gl_PointSize = uSize * aSize * uPR * (1.0 / -mv.z) * 8.0;

    vColor = col;
    vAlpha = alpha;
  }
`;

const STREAM_FRAG = /* glsl */ `
  precision mediump float;
  varying vec3  vColor;
  varying float vAlpha;
  void main() {
    vec2  d = gl_PointCoord - 0.5;
    float r = length(d);
    if (r > 0.5) discard;
    gl_FragColor = vec4(vColor, smoothstep(0.5, 0.05, r) * vAlpha);
  }
`;

export default function LeakScene({ shares }: { shares: LeakShares }) {
  const host = useRef<HTMLDivElement | null>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    const el = host.current;
    if (!el) return;

    const stage = createStage(el, { reduced, fov: 44, parallax: 0.09 });
    if (!stage) return;

    const { scene, camera, world } = stage;
    scene.fog = new THREE.FogExp2(0x05060a, 0.036);
    camera.position.set(0, 0.55, 8.4);
    camera.lookAt(0, 0.2, 0);

    const rnd = seededRandom();
    const GOLD = new THREE.Color("#e8b24c");

    /* ---- floor ---- */
    const floorMat = new THREE.ShaderMaterial({
      uniforms: { uLine: { value: new THREE.Color("#8ea3c4") }, uTime: { value: 0 } },
      vertexShader: FLOOR_VERT,
      fragmentShader: FLOOR_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, -2.35, -14);
    world.add(floor);

    // The same ruling overhead, much fainter. It closes the room, so the stream reads as
    // being *inside* something rather than floating in an unlit void.
    const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(140, 140), floorMat.clone());
    (ceiling.material as THREE.ShaderMaterial).uniforms.uLine.value = new THREE.Color("#5c6a86");
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.set(0, 5.4, -14);
    world.add(ceiling);

    /* ---- invoices ---- */
    const sheetGeo = new THREE.PlaneGeometry(1, 1.34);
    const sheets: { mesh: THREE.Mesh; phase: number; drift: number; spin: number }[] = [];
    const SHEET_COUNT = window.matchMedia("(max-width: 767px)").matches ? 9 : 18;

    for (let i = 0; i < SHEET_COUNT; i++) {
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uAccent: { value: GOLD.clone() },
          uOpacity: { value: 0.30 + rnd() * 0.42 },
          uSeed: { value: rnd() * 100 },
        },
        vertexShader: SHEET_VERT,
        fragmentShader: SHEET_FRAG,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(sheetGeo, mat);

      // Pushed to the sides and into depth so the centre band stays clear for the stream
      // and the left third stays clear for the headline.
      const side = rnd() > 0.5 ? 1 : -1;
      mesh.position.set(
        side * (4.6 + rnd() * 7.2),
        -1.8 + rnd() * 6.0,
        -4.5 - rnd() * 24,
      );
      const scale = 0.9 + rnd() * 1.9;
      mesh.scale.setScalar(scale);
      mesh.rotation.set((rnd() - 0.5) * 0.5, (rnd() - 0.5) * 1.1, (rnd() - 0.5) * 0.34);
      world.add(mesh);
      sheets.push({
        mesh,
        phase: rnd() * Math.PI * 2,
        drift: 0.1 + rnd() * 0.22,
        spin: (rnd() - 0.5) * 0.06,
      });
    }

    /* ---- the gate ---- */
    const SPAN = 17;
    const GATE = -1.4;
    const gateGroup = new THREE.Group();
    gateGroup.position.x = GATE;
    world.add(gateGroup);

    const gatePts: number[] = [0, 2.0, 0, 0, -2.0, 0];
    for (let i = -6; i <= 6; i++) {
      const y = (i / 6) * 2.0;
      gatePts.push(-0.12, y, 0, 0.12, y, 0);
    }
    const gateGeo = new THREE.BufferGeometry();
    gateGeo.setAttribute("position", new THREE.Float32BufferAttribute(gatePts, 3));
    gateGroup.add(
      new THREE.LineSegments(
        gateGeo,
        new THREE.LineBasicMaterial({ color: 0xe8b24c, transparent: true, opacity: 0.5 }),
      ),
    );

    // A soft pane of light in the gate's plane, so the split looks like it happens
    // somewhere rather than nowhere.
    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(1.5, 4.4),
      new THREE.MeshBasicMaterial({
        color: 0xe8b24c,
        transparent: true,
        opacity: 0.05,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    gateGroup.add(glow);

    /* ---- the stream ---- */
    const mobile = window.matchMedia("(max-width: 767px)").matches;
    const COUNT = mobile ? 3000 : 7000;

    const pos = new Float32Array(COUNT * 3);
    const kind = new Float32Array(COUNT);
    const speed = new Float32Array(COUNT);
    const size = new Float32Array(COUNT);

    const total = shares.valid + shares.addressed + shares.lost || 1;
    const cutValid = shares.valid / total;
    const cutAddressed = cutValid + shares.addressed / total;

    for (let i = 0; i < COUNT; i++) {
      pos[i * 3] = rnd();
      pos[i * 3 + 1] = rnd() * 2 - 1;
      pos[i * 3 + 2] = rnd() * 2 - 1;
      const roll = rnd();
      kind[i] = roll < cutValid ? 0 : roll < cutAddressed ? 1 : 2;
      speed[i] = 0.85 + rnd() * 0.5;
      size[i] = 0.55 + rnd() * 0.85;
    }

    const streamGeo = new THREE.BufferGeometry();
    streamGeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    streamGeo.setAttribute("aKind", new THREE.BufferAttribute(kind, 1));
    streamGeo.setAttribute("aSpeed", new THREE.BufferAttribute(speed, 1));
    streamGeo.setAttribute("aSize", new THREE.BufferAttribute(size, 1));

    const streamUniforms = {
      uTime: { value: reduced ? 6.2 : 0 },
      uSpan: { value: SPAN },
      uGate: { value: GATE },
      uSize: { value: mobile ? 2.8 : 3.9 },
      uPR: { value: Math.min(window.devicePixelRatio, 1.75) },
      uGold: { value: GOLD.clone() },
      // Brighter than the CSS tokens: additive blending against near-black eats saturation,
      // and slate disappears entirely at its flat-UI value.
      uSlate: { value: new THREE.Color("#8ab4dd") },
      uJade: { value: new THREE.Color("#5ad9a2") },
      uLeak: { value: new THREE.Color("#e2534a") },
    };

    const stream = new THREE.Points(
      streamGeo,
      new THREE.ShaderMaterial({
        uniforms: streamUniforms,
        vertexShader: STREAM_VERT,
        fragmentShader: STREAM_FRAG,
        transparent: true,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    stream.frustumCulled = false;
    world.add(stream);

    // The whole composition sits right of centre: the headline owns the left, so a fork
    // centred in the viewport is a fork behind the type.
    world.position.x = 2.5;

    stage.onResize((w) => {
      // Narrow viewports need the camera pulled back or the fork crops. This is a
      // recomposition, not a scale — the brief's point about art direction at breakpoints.
      camera.position.z = w < 700 ? 10.6 : w < 1100 ? 9.4 : 8.4;
      camera.position.y = w < 700 ? 0.3 : 0.55;
      world.position.x = w < 700 ? 0.6 : w < 1100 ? 1.7 : 2.5;
      camera.lookAt(0, 0.2, 0);
    });

    stage.onFrame((t) => {
      streamUniforms.uTime.value = t * 0.62;
      floorMat.uniforms.uTime.value = t;
      (ceiling.material as THREE.ShaderMaterial).uniforms.uTime.value = t * 0.6;

      for (const s of sheets) {
        s.mesh.position.y += Math.sin(t * s.drift + s.phase) * 0.0016;
        s.mesh.rotation.y += s.spin * 0.01;
        s.mesh.rotation.z = Math.sin(t * s.drift * 0.7 + s.phase) * 0.05;
      }

      glow.material.opacity = 0.04 + Math.sin(t * 1.1) * 0.015;
    });

    paintOnce(stage);

    return () => {
      sheetGeo.dispose();
      stage.dispose();
    };
  }, [reduced, shares.valid, shares.addressed, shares.lost]);

  return <div ref={host} className="absolute inset-0" aria-hidden="true" />;
}
