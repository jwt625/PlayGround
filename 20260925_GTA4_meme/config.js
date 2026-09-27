globalThis.DEFAULT_MEME = {
  title: ['grand theft', 'alignment'], subtitle: 'LIBERTY CITY → COMPUTE CITY',
  intro: 0, hold: 9, outro: 0, fade: 0.38, labels: true,
  background: 'assets/background-pioneer.png',
  backgrounds: [
    {name:'Pioneer Building · OpenAI history', image:'assets/background-pioneer.png'},
    {name:'Stargate · Abilene', image:'assets/background-stargate.png'},
    {name:'NVIDIA · Voyager & Endeavor', image:'assets/background-nvidia.png'},
    {name:'xAI · Colossus, Memphis', image:'assets/background-colossus.png'},
    {name:'Anthropic · 500 Howard', image:'assets/background-anthropic.png'},
    {name:'Meta · Menlo Park terraces', image:'assets/background-meta.png'}
  ],
  // x/y are fractions of frame width/height; zoom is relative to cover;
  // portrait height is relative to frame height. Endpoints span one card.
  motionProfiles: {
    arrival: {name:'Gentle push-in · figure up-left',
      background:{x:[-.009,.007], y:[.006,-.004], zoom:[1.065,1.09], ease:.12},
      portrait:{x:[.009,-.010], y:[.017,.006], height:[1.13,1.198], ease:.2}},
    opportunity: {name:'Slow pullback · figure down-right',
      background:{x:[.008,-.005], y:[-.004,.005], zoom:[1.095,1.07], ease:.18},
      portrait:{x:[-.009,.012], y:[.006,.019], height:[1.145,1.138], ease:.1}},
    supplier: {name:'Campus push-in · figure down-left',
      background:{x:[-.004,.006], y:[.009,-.009], zoom:[1.06,1.082], ease:.1},
      portrait:{x:[.008,-.009], y:[.007,.023], height:[1.135,1.203], ease:.16}},
    performance: {name:'Diagonal pullback · figure up-right',
      background:{x:[.010,-.009], y:[.006,-.004], zoom:[1.09,1.07], ease:.15},
      portrait:{x:[-.012,.011], y:[.025,.007], height:[1.14,1.197], ease:.12}},
    briefing: {name:'Quiet push-in · figure mostly upward',
      background:{x:[-.006,.003], y:[-.003,.006], zoom:[1.075,1.091], ease:.2},
      portrait:{x:[.002,-.003], y:[.024,.006], height:[1.14,1.186], ease:.12}},
    empire: {name:'Terrace pullback · figure down-right',
      background:{x:[.006,-.004], y:[.008,-.005], zoom:[1.10,1.073], ease:.12},
      portrait:{x:[-.005,.008], y:[.006,.025], height:[1.145,1.135], ease:.2}}
  },
  audio: 'assets/theme.m4a', audioStart: 13,
  cards: [
    {id:'ilya', name:'ILYA SUTSKEVER', role:'NIKO BELLIC', tag:'THE NEW BEGINNING', caption:'I came here for safe superintelligence.', image:'assets/ilya.png', side:1, tint:'#c5a27e'},
    {id:'sam', name:'SAM ALTMAN', role:'ROMAN BELLIC', tag:'THE BIG OPPORTUNITY', caption:'Cousin! Let’s build another gigawatt.', image:'assets/sam.png', side:-1, tint:'#a1bbc8'},
    {id:'jensen', name:'JENSEN HUANG', role:'LITTLE JACOB', tag:'THE HARDWARE CONNECTION', caption:'I got the hardware. You got the power?', image:'assets/jensen.png', side:1, tint:'#b7c98a'},
    {id:'elon', name:'ELON MUSK', role:'BRUCIE KIBBUTZ', tag:'THE PERFORMANCE ENTHUSIAST', caption:'More compute. More power. More everything.', image:'assets/elon.png', side:-1, tint:'#c6a088'},
    {id:'dario', name:'DARIO AMODEI', role:'UL PAPER · LOOSE CASTING', tag:'THE SAFETY BRIEFING', caption:'One more mission. First, the safety eval.', image:'assets/dario.png', side:1, tint:'#c3baa7'},
    {id:'mark', name:'MARK ZUCKERBERG', role:'PLAYBOY X · LOOSE CASTING', tag:'THE EMPIRE BUILDER', caption:'The weights are open. The empire is mine.', image:'assets/mark.png', side:-1, tint:'#91aace'}
  ]
};
// Explicit scene assignments also survive JSON export from the editor.
const sceneSettings = [
  ['background-pioneer','arrival'],
  ['background-stargate','opportunity'],
  ['background-nvidia','supplier'],
  ['background-colossus','performance'],
  ['background-anthropic','briefing'],
  ['background-meta','empire']
];
globalThis.DEFAULT_MEME.cards.forEach((card,i)=>{
  card.background=`assets/${sceneSettings[i][0]}.png`;
  card.motion=sceneSettings[i][1];
});
