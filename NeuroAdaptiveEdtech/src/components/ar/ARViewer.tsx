import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { ARButton, XR } from '@react-three/xr';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

// A simple 3D box component
function Box() {
    return (
        <mesh position={[0, 0, -0.5]}>
            <boxGeometry args={[0.1, 0.1, 0.1]} />
            <meshStandardMaterial color="hotpink" />
        </mesh>
    );
}

// Custom AR Button with proper typing
function CustomARButton() {
    const handleClick = async () => {
        if (!navigator.xr) {
            alert('WebXR not supported in your browser');
            return;
        }

        try {
            const session = await (navigator as any).xr.requestSession('immersive-ar', {
                requiredFeatures: ['hit-test'],
                optionalFeatures: ['dom-overlay'],
                domOverlay: { root: document.body }
            });
            console.log('AR session started:', session);
        } catch (err) {
            console.error('AR session failed:', err);
        }
    };

    return (
        <button
            onClick={handleClick}
            style={{
                position: 'absolute',
                left: '50%',
                top: '20px',
                transform: 'translateX(-50%)',
                padding: '12px 24px',
                background: 'white',
                border: '1px solid #ccc',
                borderRadius: '4px',
                cursor: 'pointer',
                zIndex: 1000,
                fontFamily: 'sans-serif',
                fontSize: '16px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
            }}
        >
            Start AR Experience
        </button>
    );
}

export default function ARViewer() {
    return (
        <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
            <CustomARButton />

            <Canvas
                camera={{ position: [0, 0, 2], fov: 50 }}
                gl={{ antialias: true }}
                dpr={[1, 2]}
            >
                <XR>
                    <Suspense fallback={null}>
                        <ambientLight intensity={0.5} />
                        <pointLight position={[10, 10, 10]} intensity={1} />
                        <Box />
                    </Suspense>
                    <OrbitControls
                        enableZoom={true}
                        enablePan={true}
                        enableRotate={true}
                    />
                </XR>
            </Canvas>
        </div>
    );
}