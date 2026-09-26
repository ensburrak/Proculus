(()=>{'use strict';
const $=s=>document.querySelector(s);
const copy={
 overview:['Gürültüyü ayır.<br><em>Bağlamı gör.</em>','PRISMA, piyasa rejimlerini, model kanıtını ve risk politikalarını tek bir karar akışında görselleştirir. Bu sayfa gerçek hesapları kontrol etmez.','İşlem kararı, farklı katmanların birbirini doğrulamasıyla anlam kazanır.'],
 core:['Kararı çöz.<br><em>Gerekçesini koru.</em>','Gözlem → rejim → model → risk sıralaması bir skorun neden tek başına yeterli olmadığını açıklar.','Bir model sinyali, yürütme ve risk kanıtı oluşmadan emir gerekçesi değildir.'],
 regime:['Piyasayı dinle.<br><em>Rejimi ayır.</em>','Trend, sıkışma, volatilite ve şok zamanları için farklı analiz düzenleri gerekir.','Şok rejiminde korelasyon ve likidite varsayımları özellikle sınanmalıdır.'],
 risk:['Önce koru.<br><em>Sonra büyü.</em>','Paper, demo ve canlı sistemler farklı kontrol kapılarına ihtiyaç duyar. Buradaki görsel kontroller gerçek limitleri değiştirmez.','Bir görsel durum işareti sunucu tarafındaki risk politikasının yerini tutmaz.'],
 research:['Deneyi belgele.<br><em>Kanıtı ayır.</em>','Backtest, örneklem dışı doğrulama, shadow ve paper sonuçları birbirinden ayrı yorumlanır.','Strateji kabulü, tek bir yüksek getiri örneğinden çıkarılamaz.'],
 settings:['Ayır. Açıkla.<br><em>Denetle.</em>','Bu sayfa yalnız görsel durumu düzenler. API anahtarını HTML dosyasına veya tarayıcı belleğine yazma.','Gerçek yapılandırma değişiklikleri kimlik doğrulamalı, denetlenebilir sunucu yolunda yapılmalıdır.']
};
let active='overview',moving=true,threeReady=false,three=null,energy=0,scene=null,root=null,renderer=null,camera=null,wing=null,wingOriginal=null,rings=[],raf=0,stream=null,analyser=null,audioCtx=null,voiceTimer=0,recognizer=null,introEl=null;
function status(message){$('#voice-status').textContent=message;}
function tip(message){$('#tip').textContent=message;}
function show(page){
 if(!copy[page])page='overview';active=page;const data=copy[page];
 $('#page-title').innerHTML=data[0];$('#page-description').textContent=data[1];tip(data[2]);
 document.querySelectorAll('[data-page]').forEach(b=>{let on=b.dataset.page===page;b.classList.toggle('on',on);b.setAttribute('aria-current',on?'page':'false')});
 if(threeReady)root.rotation.z=page==='risk'?.10:page==='regime'?-.10:0;
}
document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>show(b.dataset.page)));
$('#show-meaning').addEventListener('click',()=>{tip('Kanatlar değişen piyasa koşullarını; üç halka rejim, model ve risk kapılarını; akış kuyruğu karar geçmişini temsil eder. Bu sahne sinyal üretmez.');$('#context-title').textContent='3D nesnenin anlam haritası';});
function intro(){
 introEl?.remove();
 const el=document.createElement('div');el.className='intro';el.setAttribute('role','dialog');el.setAttribute('aria-label','Proculus PRISMA açılış animasyonu');
 el.innerHTML='<span class="eyebrow">PROCULUS / IMMERSIVE ENGINE</span><div class="sigil">◈</div><h1>PRISMA<span>.</span></h1><p>Gürültüden karara, karardan açıklamaya.</p><button id="intro-skip">STÜDYOYA GİR ↗</button>';
 document.body.appendChild(el);introEl=el;
 const end=()=>{if(!introEl||introEl!==el)return;el.classList.add('done');setTimeout(()=>el.remove(),760);introEl=null;};
 $('#intro-skip').addEventListener('click',end);
 setTimeout(end,matchMedia('(prefers-reduced-motion:reduce)').matches?140:3500);
}
$('#replay').addEventListener('click',intro);
$('#pause').addEventListener('click',()=>{moving=!moving;$('#pause').textContent=moving?'3D durdur':'3D başlat';renderOnce();});
function setEnergy(v){energy=Math.max(0,Math.min(1,v));$('#meter').style.width=(energy*100)+'%';document.documentElement.style.setProperty('--energy',String(energy));if(root)root.scale.setScalar(1+energy*.08);}
function stopSpeech(){try{speechSynthesis.cancel()}catch(_){};clearInterval(voiceTimer);voiceTimer=0;setEnergy(0)}
let speaking=false;
$('#speak').addEventListener('click',()=>{
 if(speaking){stopSpeech();speaking=false;$('#speak').textContent='♫ Anlat';status('Anlatım durduruldu.');return;}
 if(!('speechSynthesis' in window)){status('Tarayıcıda sesli anlatım yok.');return;}
 const u=new SpeechSynthesisUtterance('Prisma konuşuyor. '+$('#tip').textContent);u.lang='tr-TR';u.rate=1.04;u.pitch=1.13;
 let v=speechSynthesis.getVoices().find(v=>v.lang.toLowerCase().startsWith('tr'));if(v)u.voice=v;
 u.onstart=()=>{speaking=true;$('#speak').textContent='■ Durdur';status('PRISMA konuşuyor.');let t=0;voiceTimer=setInterval(()=>{t+=.22;setEnergy(.2+Math.abs(Math.sin(t))*.56)},76)};
 u.onend=u.onerror=()=>{speaking=false;clearInterval(voiceTimer);voiceTimer=0;setEnergy(0);$('#speak').textContent='♫ Anlat';status('Türkçe ses rehberi hazır.');};
 speechSynthesis.cancel();speechSynthesis.speak(u);
});
function phrase(text){
 const str=String(text).toLocaleLowerCase('tr-TR');
 if(str.includes('sus')||str.includes('sustur')){stopSpeech();status('Ses kapatıldı.');return;}
 const maps=[['risk','risk'],['strateji','research'],['test','research'],['veri','regime'],['rejim','regime'],['karar','core'],['ayar','settings'],['ana ekran','overview']];
 const match=maps.find(([s])=>str.includes(s));
 if(match){show(match[1]);status('Gezinme komutu uygulandı: '+match[0]);}else status('Komut anlaşılamadı. Örnek: “Risk motorunu aç”.');
}
$('#listen').addEventListener('click',()=>{
 if(recognizer){recognizer.stop();recognizer=null;status('Sesli gezinme sonlandırıldı.');return;}
 const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!SR){status('Bu tarayıcıda sesli komut tanıma bulunmuyor.');return;}
 const r=new SR();recognizer=r;r.lang='tr-TR';r.continuous=false;r.interimResults=false;
 r.onresult=e=>phrase(e.results?.[0]?.[0]?.transcript||'');
 r.onerror=e=>status('Sesli komut: '+e.error);
 r.onend=()=>{recognizer=null;$('#listen').textContent='⌘ Sesli gezinme';};
 try{r.start();$('#listen').textContent='■ Dinlemeyi bitir';status('Yalnızca gezinme dinleniyor. Bu özellik tarayıcıya göre harici tanıma servisini kullanabilir.');}catch(e){recognizer=null;status('Dinleme başlatılamadı.');}
});
function stopMic(){
 if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;
 if(audioCtx)void audioCtx.close();audioCtx=null;analyser=null;setEnergy(0);
 $('#mic').textContent='◌ Ses tepkisi';
}
$('#mic').addEventListener('click',async()=>{
 if(stream){stopMic();status('Mikrofon kapatıldı.');return;}
 if(!navigator.mediaDevices?.getUserMedia){status('Mikrofon için HTTPS veya localhost gerekli olabilir.');return;}
 try{
  stream=await navigator.mediaDevices.getUserMedia({audio:true,video:false});
  audioCtx=new (window.AudioContext||window.webkitAudioContext)();
  analyser=audioCtx.createAnalyser();analyser.fftSize=512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const data=new Uint8Array(512);
  $('#mic').textContent='■ Mikrofonu kapat';
  status('Ses seviyesi yalnız bu cihazda analiz ediliyor; bu mod ses kayıtlarını bir sunucuya göndermez.');
  const tick=()=>{if(!stream||!analyser)return;analyser.getByteTimeDomainData(data);let sum=0;for(let i=0;i<data.length;i++){let f=(data[i]-128)/128;sum+=f*f;}setEnergy(Math.min(1,Math.sqrt(sum/data.length)*6));requestAnimationFrame(tick);};
  tick();
 }catch(e){status('Mikrofona erişilemedi: '+e.message);}
});
function renderOnce(){if(renderer&&scene&&camera)renderer.render(scene,camera);}
async function init3D(){
 const host=$('#three'),url='https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js';
 try{
  const T=await import(/* @vite-ignore */ url);
  three=T;scene=new T.Scene();camera=new T.PerspectiveCamera(41,1,.1,50);camera.position.set(0,.32,7.2);
  renderer=new T.WebGLRenderer({alpha:true,antialias:true,powerPreference:'high-performance'});
  renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.18;
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.6));host.appendChild(renderer.domElement);
  const ambient=new T.AmbientLight(0xbeb1ff,1.7);scene.add(ambient);
  const light=new T.PointLight(0xc4baff,5,15,1);light.position.set(-3,3,4);scene.add(light);
  const light2=new T.PointLight(0x68dfff,3.6,15,1);light2.position.set(3,-2,2);scene.add(light2);
  root=new T.Group();scene.add(root);
  const shine=new T.MeshPhysicalMaterial({color:0xeeeaff,metalness:.22,roughness:.16,clearcoat:1});
  const dark=new T.MeshPhysicalMaterial({color:0x171b3c,metalness:.5,roughness:.24});
  const irid=new T.MeshPhysicalMaterial({vertexColors:true,metalness:.25,roughness:.16,clearcoat:1,side:T.DoubleSide,transparent:true,opacity:.97});
  const ball=(r,m,xyz,scale=[1,1,1])=>{let o=new T.Mesh(new T.SphereGeometry(r,30,24),m);o.position.set(...xyz);o.scale.set(...scale);root.add(o);return o;};
  // A curved procedural manta. Vertebrae deform by time: physical signal-regime metaphor.
  const nx=41,nz=28,p=[],colors=[],idx=[],a=new T.Color(0xc5a4ff),b=new T.Color(0x67dcf7);
  for(let z=0;z<=nz;z++){let v=z/nz;for(let x=0;x<=nx;x++){let u=x/nx*2-1;let taper=(1-v*.69)*Math.pow(Math.max(.1,1-u*u*.5),.86);
    p.push(u*1.9*taper,.05+Math.sin(v*Math.PI)*.37-Math.abs(u)*.37,(v-.43)*1.52);
    let cc=a.clone().lerp(b,.20+.58*v+.15*Math.abs(u));colors.push(cc.r,cc.g,cc.b);
  }}
  for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){let k=z*(nx+1)+x,m=k+nx+1;idx.push(k,m,k+1,m,m+1,k+1);}
  const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(p,3));geo.setAttribute('color',new T.Float32BufferAttribute(colors,3));geo.setIndex(idx);geo.computeVertexNormals();
  wingOriginal=new Float32Array(p);wing=new T.Mesh(geo,irid);root.add(wing);
  ball(.58,shine,[0,.06,.37],[.76,.31,.80]);
  [-1,1].forEach(d=>{ball(.084,dark,[d*.27,.13,.75],[1,1,.55]);ball(.033,shine,[d*.28,.15,.79]);const curve=new T.CatmullRomCurve3([new T.Vector3(d*.3,-.04,-.3),new T.Vector3(d*.21,-.10,-.7),new T.Vector3(d*.14,-.2,-1.30),new T.Vector3(d*.1,-.27,-2.01)]);root.add(new T.Mesh(new T.TubeGeometry(curve,25,.023,7,false),shine));});
  for(let i=0;i<3;i++){let ring=new T.Mesh(new T.TorusGeometry(1.68+i*.28,.022,9,116),new T.MeshBasicMaterial({color:i===1?0x78dcff:0xc3aaff,transparent:true,opacity:.32-i*.06,depthWrite:false}));ring.rotation.set(.95+i*.38,.3+i*.24,.1+i*.2);ring.userData.start=ring.rotation.clone();root.add(ring);rings.push(ring);}
  let pts=new Float32Array(180*3);for(let k=0;k<180;k++){let u=k*2.399,z=1-2*(k+.5)/180,r=Math.sqrt(1-z*z),dist=2.6+k%8*.12;pts[k*3]=Math.cos(u)*r*dist;pts[k*3+1]=z*dist;pts[k*3+2]=Math.sin(u)*r*dist;}
  let particles=new T.BufferGeometry();particles.setAttribute('position',new T.BufferAttribute(pts,3));root.add(new T.Points(particles,new T.PointsMaterial({color:0xb9a8ff,size:.021,opacity:.55,transparent:true})));
  function size(){let w=Math.max(250,host.clientWidth),h=Math.max(270,host.clientHeight);renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
  if(window.ResizeObserver)new ResizeObserver(size).observe(host);else window.addEventListener('resize',size);
  size();
  let pointer={x:0,y:0,drag:0,last:0};
  renderer.domElement.addEventListener('pointerdown',e=>{pointer.drag=1;pointer.last=e.clientX;renderer.domElement.setPointerCapture?.(e.pointerId)});
  renderer.domElement.addEventListener('pointermove',e=>{const r=renderer.domElement.getBoundingClientRect();pointer.x=((e.clientX-r.left)/r.width-.5)*2;pointer.y=((e.clientY-r.top)/r.height-.5)*2;if(pointer.drag){root.rotation.y+=(e.clientX-pointer.last)*.007;pointer.last=e.clientX;}});
  renderer.domElement.addEventListener('pointerup',()=>pointer.drag=0);
  threeReady=true;$('#scene-status').textContent='THREE.JS · GERÇEK 3D';$('#visual-mode').textContent='Three.js / WebGL';
  const begin=performance.now();let last=0;
  const loop=now=>{raf=requestAnimationFrame(loop);if(document.hidden||!moving)return;if(now-last<28)return;last=now;
    const t=(now-begin)*.001,at=wing.geometry.attributes.position;
    for(let z=0;z<=nz;z++)for(let x=0;x<=nx;x++){const k=z*(nx+1)+x,u=x/nx*2-1,v=z/nz;at.array[k*3+1]=wingOriginal[k*3+1]+Math.pow(Math.abs(u),1.9)*Math.sin(t*1.9+v*3.3)*.21+Math.cos(t*.85+u*2.8)*.02;}
    at.needsUpdate=true;wing.geometry.computeVertexNormals();
    root.rotation.y+=(pointer.x*.17+Math.sin(t*.25)*.12-root.rotation.y)*.02;root.rotation.x+=(-pointer.y*.08-root.rotation.x)*.025;
    root.position.y=Math.sin(t*.9)*.065;rings.forEach((r,i)=>{r.rotation.y+=i%2?-.0014:.0017;r.rotation.x+=Math.sin(t*.35+i)*.0007;});
    renderer.render(scene,camera);
  };
  raf=requestAnimationFrame(loop);
 }catch(err){$('#scene-status').textContent='CSS 3D YEDEK · THREE.JS CDN GEREKLİ';$('#visual-mode').textContent='CSS / Three.js bekleniyor';console.info('Three.js modülü erişilemedi.',err);}
}
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopSpeech();if(recognizer)recognizer.stop();if(stream)stopMic();}});
window.addEventListener('pagehide',()=>{stopSpeech();if(recognizer)recognizer.stop();stopMic();cancelAnimationFrame(raf);renderer?.dispose();});
show('overview');intro();void init3D();
})();