import * as THREE from 'three';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

const w_MMI = 6
const L_MMI = 15
const t_Si = 0.5
const geometry_cube = new THREE.BoxGeometry( w_MMI, L_MMI, t_Si );
const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const cube = new THREE.Mesh( geometry_cube, material );
scene.add( cube );


const length = 6, width = 2, d_taper = 3, w_WG = 1;

const shape = new THREE.Shape();
shape.moveTo( d_taper/2 - width/2, L_MMI/2 );
shape.lineTo( d_taper/2 + width/2, L_MMI/2 );
shape.lineTo( d_taper/2 + w_WG/2, L_MMI/2 + length );
shape.lineTo( d_taper/2 - w_WG/2, L_MMI/2 + length );
shape.lineTo( d_taper/2 - width/2, L_MMI/2 );
shape.lineTo( -d_taper/2 - width/2, L_MMI/2 );
shape.lineTo( -d_taper/2 + width/2, L_MMI/2 );
shape.lineTo( -d_taper/2 + w_WG/2, L_MMI/2 + length );
shape.lineTo( -d_taper/2 - w_WG/2, L_MMI/2 + length );
shape.lineTo( -d_taper/2 - width/2, L_MMI/2 );

const extrudeSettings = {
	steps: 1,
	depth: t_Si,
	bevelEnabled: true,
	bevelThickness: 0.1,
	bevelSize: 0.1,
	bevelOffset: 0,
	bevelSegments: 1
};

const geometry = new THREE.ExtrudeGeometry( shape, extrudeSettings );
// const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const mesh = new THREE.Mesh( geometry, material ) ;
scene.add( mesh );


const shape_taper_in = new THREE.Shape();
shape_taper_in.moveTo( - width/2, -L_MMI/2 );
shape_taper_in.lineTo(  + width/2, -L_MMI/2 );
shape_taper_in.lineTo(  + w_WG/2, -L_MMI/2 - length );
shape_taper_in.lineTo(  - w_WG/2, -L_MMI/2 - length );
shape_taper_in.lineTo(  - width/2, -L_MMI/2 );
const geometry_taper_in = new THREE.ExtrudeGeometry( shape_taper_in, extrudeSettings );
// const material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
const mesh_taper_in = new THREE.Mesh( geometry_taper_in, material ) ;
scene.add( mesh_taper_in );


camera.position.z = 40;

function animate() {
	requestAnimationFrame( animate );
    mesh.rotation.x += 0.01;
    mesh.rotation.y += 0.01;
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    mesh_taper_in.rotation.x += 0.01;
    mesh_taper_in.rotation.y += 0.01;
	renderer.render( scene, camera );
}
animate();