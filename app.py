import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.sparse as sp

st.set_page_config(
    page_title="IMDB Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
#  WARM AUTUMN GLOW  ·  #003049 · #D62828 · #F77F00 · #FCBF49 · #EAE2B7
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── PALETTE ─────────────────────────────────────────── */
:root {
  --navy:   #003049;
  --red:    #D62828;
  --orange: #F77F00;
  --amber:  #FCBF49;
  --cream:  #EAE2B7;

  --bg:     #01090f;
  --bg2:    #020e18;
  --card:   #040f1a;
  --border: rgba(247,127,0,0.18);
  --text:   #EAE2B7;
  --muted:  rgba(234,226,183,0.48);
}

/* ── BASE ────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
  font-family: 'Rajdhani', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
  cursor: none !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 1400px; }
p, li, .stMarkdown { color: var(--cream); }

/* ── CURSOR ─────────────────────────────────────────── */
#c-dot {
  position: fixed; pointer-events: none; z-index: 99999;
  width: 10px; height: 10px; background: var(--amber); border-radius: 50%;
  transform: translate(-50%,-50%);
  box-shadow: 0 0 14px var(--amber), 0 0 40px rgba(252,191,73,.55);
  transition: width .15s, height .15s, background .2s, box-shadow .2s;
}
#c-ring {
  position: fixed; pointer-events: none; z-index: 99998;
  width: 40px; height: 40px;
  border: 2px solid rgba(247,127,0,.65); border-radius: 50%;
  transform: translate(-50%,-50%);
  transition: width .22s ease, height .22s ease, border-color .22s;
}
#c-trail {
  position: fixed; pointer-events: none; z-index: 99997;
}
#c-glow {
  position: fixed; pointer-events: none; z-index: 2;
  width: 460px; height: 460px; border-radius: 50%;
  background: radial-gradient(circle, rgba(247,127,0,.07) 0%, rgba(214,40,40,.03) 45%, transparent 70%);
  transform: translate(-50%,-50%);
  transition: left .3s ease, top .3s ease;
}

/* ── STAR CANVAS ────────────────────────────────────── */
#bg-canvas {
  position: fixed; top:0; left:0;
  width:100%; height:100%;
  pointer-events:none; z-index:0;
}

/* ── PARTICLES ──────────────────────────────────────── */
.ptcl-wrap {
  position: fixed; top:0; left:0;
  width:100%; height:100%;
  pointer-events:none; z-index:1; overflow:hidden;
}
.pt {
  position:absolute; border-radius:50%;
  animation: fup linear infinite; opacity:0;
}
@keyframes fup {
  0%   { transform:translateY(110vh) scale(0); opacity:0; }
  8%   { opacity:.6; }
  92%  { opacity:.25; }
  100% { transform:translateY(-10vh) scale(1.4); opacity:0; }
}

/* ── CONTENT ABOVE CANVAS ───────────────────────────── */
.stApp > div { position:relative; z-index:10; }

/* ── HERO ───────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, #010c14 0%, #021525 55%, #010c14 100%);
  border: 1px solid rgba(247,127,0,.22);
  border-radius: 24px; padding: 52px 66px;
  margin-bottom: 24px; position:relative; overflow:hidden;
}
.hero::before {
  content:''; position:absolute; inset:0;
  background: linear-gradient(108deg, transparent 28%, rgba(252,191,73,.04) 50%, transparent 72%);
  animation: scan 5s ease-in-out infinite;
}
@keyframes scan {
  0%,100% { transform:translateX(-110%); }
  50%      { transform:translateX(110%); }
}
.hero::after {
  content:'🎞'; position:absolute;
  right:60px; top:50%; transform:translateY(-50%) rotate(-8deg);
  font-size:120px; opacity:.05;
  animation: reel 14s linear infinite;
}
@keyframes reel { to { transform:translateY(-50%) rotate(352deg); } }

.hero-title {
  font-family:'Bebas Neue',sans-serif;
  font-size:5rem; letter-spacing:6px; line-height:1;
  background: linear-gradient(135deg, var(--amber) 0%, var(--orange) 48%, var(--red) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation: slideL .75s ease both .1s;
}
@keyframes slideL {
  from { transform:translateX(-42px); opacity:0; }
  to   { transform:translateX(0);     opacity:1; }
}

/* ── METRIC CARDS ───────────────────────────────────── */
.mcg { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }

.mc {
  background: linear-gradient(145deg, #040f1a, #061828);
  border: 1px solid rgba(247,127,0,.2);
  border-radius: 18px; padding:28px 18px; text-align:center;
  position:relative; overflow:hidden;
  transition: transform .35s ease, box-shadow .35s, border-color .25s;
  animation: cardUp .6s ease both;
  transform-style: preserve-3d;
}
.mc:nth-child(1){animation-delay:.06s}
.mc:nth-child(2){animation-delay:.13s}
.mc:nth-child(3){animation-delay:.20s}
.mc:nth-child(4){animation-delay:.27s}
@keyframes cardUp {
  from { transform:translateY(30px) scale(.9); opacity:0; }
  to   { transform:translateY(0) scale(1);     opacity:1; }
}
.mc::before {
  content:''; position:absolute;
  left:0; top:18%; height:64%; width:3px;
  background: linear-gradient(180deg, var(--orange), var(--red));
  border-radius:0 3px 3px 0; opacity:.75;
}
.mc::after {
  content:''; position:absolute; top:0; right:0;
  width:50px; height:50px;
  background: linear-gradient(225deg, rgba(247,127,0,.2), transparent);
  border-radius:0 18px 0 50px;
}
.mc:hover {
  transform: translateY(-10px) rotateX(7deg) rotateY(-4deg) scale(1.05);
  box-shadow: 0 24px 55px rgba(247,127,0,.2), 0 0 0 1px rgba(247,127,0,.38);
  border-color: rgba(247,127,0,.48);
}

.mc-ico { font-size:1.75rem; margin-bottom:8px; }
.mc-val {
  font-family:'Bebas Neue',sans-serif; font-size:2.6rem; letter-spacing:2px;
  background: linear-gradient(135deg, var(--amber), var(--orange));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.mc-lbl { font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px; margin-top:5px; }

/* ── SECTION HEADING ────────────────────────────────── */
.sh {
  font-family:'Bebas Neue',sans-serif; font-size:1.55rem; letter-spacing:4px;
  color:var(--amber); border-left:4px solid var(--red);
  padding-left:14px; margin:24px 0 14px; position:relative;
}
.sh::after {
  content:''; position:absolute;
  bottom:-5px; left:18px; width:55px; height:1px;
  background: linear-gradient(90deg, var(--orange), transparent);
}

/* ── TABS ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background:transparent; gap:6px;
  border-bottom:1px solid rgba(247,127,0,.2); padding-bottom:4px;
}
.stTabs [data-baseweb="tab"] {
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.06);
  border-radius:10px 10px 0 0;
  color:var(--muted);
  font-family:'Rajdhani',sans-serif; font-weight:600; font-size:1rem; letter-spacing:1.5px;
  padding:10px 28px; transition:all .25s;
}
.stTabs [data-baseweb="tab"]:hover { background:rgba(247,127,0,.1); color:var(--amber); }
.stTabs [aria-selected="true"] {
  background:linear-gradient(135deg,rgba(247,127,0,.22),rgba(214,40,40,.1)) !important;
  border-color:rgba(247,127,0,.48) !important;
  color:var(--amber) !important;
}

/* ── FORM ───────────────────────────────────────────── */
.stSelectbox>div>div,
.stTextArea>div>div>textarea {
  background:rgba(3,14,25,.85) !important;
  border:1px solid rgba(247,127,0,.22) !important;
  border-radius:10px !important; color:var(--text) !important;
  font-family:'Rajdhani',sans-serif !important; font-size:1rem !important;
  transition:border-color .3s, box-shadow .3s !important;
}
.stSelectbox>div>div:hover,
.stTextArea>div>div>textarea:focus {
  border-color:rgba(247,127,0,.65) !important;
  box-shadow:0 0 22px rgba(247,127,0,.14) !important;
}
label, .stSlider label {
  color:var(--muted) !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:.74rem !important; text-transform:uppercase !important; letter-spacing:1.5px !important;
}
[data-baseweb="slider"] div[role="slider"] { background:var(--orange) !important; }

/* ── BUTTON ─────────────────────────────────────────── */
.stButton>button {
  background:linear-gradient(135deg, var(--red), #961e1e) !important;
  color:var(--cream) !important;
  font-family:'Bebas Neue',sans-serif !important; font-size:1.35rem !important;
  letter-spacing:5px !important; border:none !important;
  border-radius:12px !important; padding:18px 36px !important;
  width:100% !important; overflow:hidden;
  transition:transform .3s, box-shadow .3s !important;
}
.stButton>button:hover {
  transform:translateY(-5px) scale(1.02) !important;
  box-shadow:0 18px 48px rgba(214,40,40,.55), 0 0 0 1px rgba(214,40,40,.7) !important;
}
.stButton>button:active { transform:translateY(1px) scale(.98) !important; }

/* ── RESULT BOX ─────────────────────────────────────── */
.rbox {
  background:linear-gradient(145deg,#040f1a,#062040);
  border:2px solid rgba(252,191,73,.3);
  border-radius:22px; padding:38px 28px; text-align:center;
  position:relative; overflow:hidden;
  animation:rPop .52s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes rPop {
  from { transform:scale(.82) rotateX(18deg); opacity:0; }
  to   { transform:scale(1)   rotateX(0);     opacity:1; }
}
.rbox::before {
  content:''; position:absolute; top:-50%; left:-50%;
  width:200%; height:200%;
  background:conic-gradient(from 0deg, transparent, rgba(247,127,0,.04), transparent);
  animation:cspin 10s linear infinite;
}
@keyframes cspin { to { transform:rotate(360deg); } }

.r-num {
  font-family:'Bebas Neue',sans-serif; font-size:8rem; line-height:1;
  background:linear-gradient(135deg, var(--amber) 0%, var(--orange) 55%, var(--red) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  filter:drop-shadow(0 0 28px rgba(252,191,73,.45));
}
.r-stars  { font-size:2rem; letter-spacing:5px; margin:8px 0; }
.r-verdict{ font-family:'Rajdhani',sans-serif; font-size:1.35rem; font-weight:700; letter-spacing:3px; }
.r-mini   { display:flex; justify-content:space-around; margin-top:18px; border-top:1px solid rgba(252,191,73,.15); padding-top:16px; }
.r-mv     { font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:var(--amber); }
.r-ml     { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }

/* ── PLACEHOLDER ────────────────────────────────────── */
.ph {
  background:rgba(255,255,255,.02);
  border:1px dashed rgba(247,127,0,.18); border-radius:20px;
  padding:60px 30px; text-align:center;
  animation:breathe 3s ease-in-out infinite;
}
@keyframes breathe {
  0%,100% { box-shadow:0 0 16px rgba(247,127,0,.02); border-color:rgba(247,127,0,.12); }
  50%      { box-shadow:0 0 40px rgba(247,127,0,.08); border-color:rgba(247,127,0,.28); }
}
.ph-ico { font-size:4rem; animation:flt 3.5s ease-in-out infinite; }
@keyframes flt { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-14px); } }

/* ── DATAFRAME ──────────────────────────────────────── */
.stDataFrame { border-radius:12px; overflow:hidden; border:1px solid var(--border); }

/* ── FOOTER ─────────────────────────────────────────── */
.ft {
  text-align:center; padding:28px 0 10px;
  color:var(--muted); font-size:.78rem;
  font-family:'JetBrains Mono',monospace; letter-spacing:1px;
  border-top:1px solid rgba(255,255,255,.04); margin-top:38px;
}
</style>

<!-- ══ CANVAS ═════════════════════════════════════════════ -->
<canvas id="bg-canvas"></canvas>
<!-- ══ CURSOR ═════════════════════════════════════════════ -->
<div id="c-glow"></div>
<div id="c-dot"></div>
<div id="c-ring"></div>
<!-- ══ PARTICLES ══════════════════════════════════════════ -->
<div class="ptcl-wrap" id="ptcl"></div>

<script>
/* ── STARFIELD ───────────────────────────────────────── */
(function(){
  const c=document.getElementById('bg-canvas');
  const ctx=c.getContext('2d');
  let W,H,stars=[];
  const resize=()=>{ W=c.width=window.innerWidth; H=c.height=window.innerHeight; };
  resize(); window.addEventListener('resize',resize);
  const cols=['#FCBF49','#F77F00','#D62828','#EAE2B7','#ffffff'];
  for(let i=0;i<260;i++) stars.push({
    x:Math.random()*2400-1200, y:Math.random()*2400-1200,
    z:Math.random()*1000, sz:Math.random()*1.8+.3,
    spd:Math.random()*.45+.12, col:cols[Math.floor(Math.random()*cols.length)]
  });
  let mx=0,my=0;
  document.addEventListener('mousemove',e=>{mx=e.clientX-W/2;my=e.clientY-H/2;});
  function draw(){
    ctx.clearRect(0,0,W,H);
    stars.forEach(s=>{
      s.z-=s.spd; if(s.z<=0)s.z=1000;
      const px=(s.x+mx*.018)/s.z*W/2+W/2;
      const py=(s.y+my*.018)/s.z*H/2+H/2;
      const r=s.sz*(1-s.z/1000)*2.8;
      const a=(1-s.z/1000)*.9;
      ctx.beginPath(); ctx.arc(px,py,r,0,Math.PI*2);
      ctx.fillStyle=s.col; ctx.globalAlpha=a;
      ctx.fill();
      if(r>1.1){ ctx.shadowBlur=8; ctx.shadowColor=s.col; ctx.fill(); ctx.shadowBlur=0; }
    });
    ctx.globalAlpha=1;
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── FLOATING PARTICLES ──────────────────────────────── */
(function(){
  const w=document.getElementById('ptcl');
  const cs=['#FCBF49','#F77F00','#D62828','#003049','#EAE2B7'];
  for(let i=0;i<28;i++){
    const p=document.createElement('div');
    p.className='pt';
    const s=Math.random()*4+1.2;
    p.style.cssText=`left:${Math.random()*100}%;width:${s}px;height:${s}px;background:${cs[Math.floor(Math.random()*cs.length)]};animation-duration:${Math.random()*14+9}s;animation-delay:${Math.random()*12}s;box-shadow:0 0 ${s*3}px currentColor;`;
    w.appendChild(p);
  }
})();

/* ── CURSOR ──────────────────────────────────────────── */
(function(){
  const dot=document.getElementById('c-dot');
  const ring=document.getElementById('c-ring');
  const glow=document.getElementById('c-glow');
  let mx=0,my=0,rx=0,ry=0,lx=0,ly=0,lt=0;

  document.addEventListener('mousemove',e=>{
    mx=e.clientX; my=e.clientY;
    dot.style.left=mx+'px'; dot.style.top=my+'px';
    glow.style.left=mx+'px'; glow.style.top=my+'px';
    const now=Date.now(),dx=mx-lx,dy=my-ly;
    const spd=Math.sqrt(dx*dx+dy*dy)/(now-lt+1)*16;
    if(spd>5&&now-lt>28){ spawnSpark(mx,my); lx=mx;ly=my;lt=now; }
  });

  function spawnSpark(x,y){
    const s=document.createElement('div');
    const cs=['#FCBF49','#F77F00','#D62828','#EAE2B7','#fff'];
    const col=cs[Math.floor(Math.random()*cs.length)];
    const vx=(Math.random()-.5)*6, vy=(Math.random()-.5)*6-2.5;
    let life=1,ax=x,ay=y;
    s.style.cssText=`position:fixed;width:4px;height:4px;border-radius:50%;background:${col};box-shadow:0 0 8px ${col};pointer-events:none;z-index:99996;left:${x}px;top:${y}px;`;
    document.body.appendChild(s);
    (function anim(){
      ax+=vx*(life*.9); ay+=vy*(life*.9)+.35;
      life-=.055;
      s.style.left=ax+'px'; s.style.top=ay+'px';
      s.style.opacity=life; s.style.transform=`scale(${life*1.6})`;
      if(life>0) requestAnimationFrame(anim); else s.remove();
    })();
  }

  /* smooth ring follow */
  (function animRing(){
    rx+=(mx-rx)*.13; ry+=(my-ry)*.13;
    ring.style.left=rx+'px'; ring.style.top=ry+'px';
    requestAnimationFrame(animRing);
  })();

  /* hover states */
  document.addEventListener('mouseover',e=>{
    const btn=e.target.closest('button');
    if(btn){
      dot.style.cssText+=';width:14px;height:14px;background:#D62828;box-shadow:0 0 20px #D62828,0 0 50px rgba(214,40,40,.45);';
      ring.style.width='58px'; ring.style.height='58px'; ring.style.borderColor='rgba(214,40,40,.8)';
    } else {
      dot.style.cssText+=';width:10px;height:10px;background:#FCBF49;box-shadow:0 0 14px #FCBF49,0 0 40px rgba(252,191,73,.55);';
      ring.style.width='40px'; ring.style.height='40px'; ring.style.borderColor='rgba(247,127,0,.65)';
    }
  });

  /* click ripple */
  document.addEventListener('click',e=>{
    const r=document.createElement('div');
    r.style.cssText=`position:fixed;left:${e.clientX}px;top:${e.clientY}px;width:4px;height:4px;border:2px solid #FCBF49;border-radius:50%;pointer-events:none;z-index:99990;transform:translate(-50%,-50%);animation:cripple .65s ease-out forwards;`;
    document.body.appendChild(r); setTimeout(()=>r.remove(),700);
  });

  const ks=document.createElement('style');
  ks.textContent='@keyframes cripple{to{width:64px;height:64px;opacity:0;border-color:rgba(252,191,73,0);}}';
  document.head.appendChild(ks);
})();

/* ── 3D CARD TILT ────────────────────────────────────── */
document.addEventListener('mousemove',e=>{
  document.querySelectorAll('.mc').forEach(card=>{
    const rc=card.getBoundingClientRect();
    const cx=rc.left+rc.width/2, cy=rc.top+rc.height/2;
    const dx=(e.clientX-cx)/rc.width, dy=(e.clientY-cy)/rc.height;
    const d=Math.sqrt(dx*dx+dy*dy);
    if(d<1.3){
      card.style.transform=`translateY(-10px) rotateX(${-dy*13}deg) rotateY(${dx*13}deg) scale(1.05)`;
      card.style.boxShadow=`0 24px 55px rgba(247,127,0,.2),0 0 0 1px rgba(247,127,0,.38)`;
    } else {
      card.style.transform=''; card.style.boxShadow='';
    }
  });
});
</script>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  DATA & MODEL
# ═══════════════════════════════════════════════════════
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 1000
    genres = ['Action','Comedy','Drama','Horror','Romance','Thriller','Sci-Fi','Animation','Documentary','Crime']
    pos = ['brilliant','masterpiece','excellent','outstanding','wonderful','fantastic','amazing','superb','incredible','perfect']
    neg = ['boring','terrible','awful','dull','disappointing','mediocre','slow','predictable','weak','bad']
    neu = ['interesting','decent','average','fine','okay','watchable','standard','typical','simple','regular']

    def gen_review(r):
        if   r>=7.5: w=np.random.choice(pos,5,True).tolist()+np.random.choice(neu,2,True).tolist()
        elif r>=5.5: w=np.random.choice(neu,4,True).tolist()+np.random.choice(pos,2,True).tolist()+np.random.choice(neg,1,True).tolist()
        else:        w=np.random.choice(neg,5,True).tolist()+np.random.choice(neu,2,True).tolist()
        t=[f"This movie is {w[0]} and {w[1]}. The story is {w[2]}. {w[3]} performance.",
           f"A {w[0]} film with {w[1]} direction. The plot is {w[2]} and {w[3]}.",
           f"Watched this movie. It is {w[0]}, {w[1]}, {w[2]}. Overall {w[3]} experience."]
        return np.random.choice(t)

    ratings = np.clip(np.random.normal(6.5,1.5,n),1.0,10.0).round(1)
    return pd.DataFrame({
        'title':          [f"Movie_{i}" for i in range(1,n+1)],
        'genre':          np.random.choice(genres,n),
        'release_year':   np.random.randint(1990,2024,n),
        'duration_min':   np.random.randint(70,210,n),
        'budget_million': np.random.uniform(1,300,n).round(2),
        'num_votes':      np.random.randint(1000,500000,n),
        'director_score': np.random.uniform(1,10,n).round(1),
        'cast_score':     np.random.uniform(1,10,n).round(1),
        'review':         [gen_review(r) for r in ratings],
        'rating':         ratings
    })

@st.cache_resource
def train_models(df):
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1,2), stop_words='english')
    tv    = tfidf.fit_transform(df['review'])
    gd    = pd.get_dummies(df['genre'], prefix='genre')
    nf    = ['duration_min','budget_million','num_votes','director_score','cast_score','release_year']
    sc    = StandardScaler()
    Xn    = sc.fit_transform(df[nf].values)
    X     = sp.hstack([sp.csr_matrix(Xn), sp.csr_matrix(gd.values), tv])
    y     = df['rating'].values
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    models = {
        'Linear Regression':  LinearRegression(),
        'Random Forest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':  GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    res={}
    for nm,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        res[nm]={'model':m,'preds':p,
                 'MAE':mean_absolute_error(yte,p),
                 'RMSE':np.sqrt(mean_squared_error(yte,p)),
                 'R2':r2_score(yte,p)}
    return res,tfidf,sc,gd.columns.tolist(),list(df['genre'].unique()),yte,Xte,nf

def do_predict(rev,genre,dur,bud,votes,dsc,csc,yr,tfidf,sc,gcols,model,nf):
    tv = tfidf.transform([rev])
    gv = np.zeros((1,len(gcols))); col=f"genre_{genre}"
    if col in gcols: gv[0,gcols.index(col)]=1
    nv = sc.transform([[dur,bud,votes,dsc,csc,yr]])
    X  = sp.hstack([sp.csr_matrix(nv),sp.csr_matrix(gv),tv])
    return float(model.predict(X)[0])

def get_stars(r):
    f=int(r/2); h=1 if(r/2-f)>=.5 else 0; e=5-f-h
    return "⭐"*f+("✨"*h)+("☆"*e)

def get_verdict(r):
    if r>=8.5: return "🏆 MASTERPIECE","#FCBF49"
    if r>=7.5: return "🎖️ EXCELLENT","#F77F00"
    if r>=6.5: return "👍 GOOD MOVIE","#6ee7b7"
    if r>=5.5: return "😐 AVERAGE","#fbbf24"
    if r>=4.0: return "👎 BELOW AVG","#fb923c"
    return "💀 POOR","#D62828"

# ── Load ───────────────────────────────────────────────
df = generate_dataset()
res,tfidf,sc,gcols,all_g,yte,Xte,nf = train_models(df)
best_nm = max(res,key=lambda x:res[x]['R2'])

# ═══════════════════════════════════════════════════════
#  HERO  (no pill, no subtitle)
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">IMDB MOVIE RATING PREDICTOR</div>
</div>
""", unsafe_allow_html=True)

# ─── METRIC CARDS ─────────────────────────────────────
br2=res[best_nm]['R2']; bm=res[best_nm]['MAE']
st.markdown(f"""
<div class="mcg">
  <div class="mc"><div class="mc-ico">🎬</div><div class="mc-val">1,000</div><div class="mc-lbl">Movies in Dataset</div></div>
  <div class="mc"><div class="mc-ico">🤖</div><div class="mc-val">3</div><div class="mc-lbl">ML Models Trained</div></div>
  <div class="mc"><div class="mc-ico">🎯</div><div class="mc-val">{br2:.3f}</div><div class="mc-lbl">Best R² Score</div></div>
  <div class="mc"><div class="mc-ico">⚡</div><div class="mc-val">±{bm:.2f}</div><div class="mc-lbl">Avg Error (MAE)</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════
t1, t2, t3 = st.tabs(["🎯  PREDICT RATING", "📊  EDA & ANALYSIS", "🤖  MODEL RESULTS"])

# ──────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ──────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="sh">ENTER MOVIE DETAILS</div>', unsafe_allow_html=True)
    cf, cr = st.columns([1.2, 1])

    with cf:
        genre = st.selectbox("🎭 Genre", sorted(all_g))
        c1, c2 = st.columns(2)
        with c1:
            year  = st.slider("📅 Release Year", 1990, 2024, 2022)
            dur   = st.slider("⏱️ Duration (min)", 70, 210, 120)
            dsc   = st.slider("🎬 Director Score", 1.0, 10.0, 7.5, 0.1)
        with c2:
            bud   = st.slider("💰 Budget ($M)", 1.0, 300.0, 80.0, 1.0)
            votes = st.slider("🗳️ Votes", 1000, 500000, 100000, 1000)
            csc   = st.slider("⭐ Cast Score", 1.0, 10.0, 7.5, 0.1)
        rev = st.text_area("📝 Write a Review",
                "This film is absolutely brilliant and captivating. Outstanding performances and excellent direction.",
                height=100)
        mdl = st.selectbox("🤖 Select ML Model", list(res.keys()))
        btn = st.button("🎬  PREDICT RATING  🎬")

    with cr:
        if btn:
            raw  = do_predict(rev,genre,dur,bud,votes,dsc,csc,year,tfidf,sc,gcols,res[mdl]['model'],nf)
            pred = round(np.clip(raw,1.0,10.0),1)
            strs = get_stars(pred); verd,vcol = get_verdict(pred)
            st.markdown(f"""
            <div class="rbox">
              <div style="color:var(--muted);font-size:.75rem;font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Predicted IMDB Rating</div>
              <div class="r-num">{pred}</div>
              <div class="r-stars">{strs}</div>
              <div class="r-verdict" style="color:{vcol}">{verd}</div>
              <div style="color:var(--muted);font-size:.78rem;margin-top:6px;font-family:'JetBrains Mono',monospace">via {mdl}</div>
              <div class="r-mini">
                <div><div class="r-mv">{dsc}</div><div class="r-ml">Director</div></div>
                <div><div class="r-mv">{csc}</div><div class="r-ml">Cast</div></div>
                <div><div class="r-mv">${bud:.0f}M</div><div class="r-ml">Budget</div></div>
                <div><div class="r-mv">{year}</div><div class="r-ml">Year</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ph">
              <div class="ph-ico">🎬</div>
              <div style="color:var(--muted);margin-top:16px;font-family:'Rajdhani';font-size:1.05rem">
                Fill in the details and click<br>
                <strong style="color:var(--amber);letter-spacing:2px">PREDICT RATING</strong>
              </div>
            </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# TAB 2 — EDA
# ──────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="sh">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

    BG='#01090f'; CARD='#040f1a'
    AMBER='#FCBF49'; ORANGE='#F77F00'; RED='#D62828'; NAVY='#003049'; CREAM='#EAE2B7'
    plt.style.use('dark_background')

    # Chart 1 — distribution + genre
    fig, axes = plt.subplots(1,2,figsize=(13,4))
    fig.patch.set_facecolor(BG)
    for ax in axes: ax.set_facecolor(CARD)

    _, bins, patches = axes[0].hist(df['rating'],bins=20,edgecolor=BG,linewidth=.5)
    cmap = plt.cm.get_cmap('YlOrRd')
    for i,p in enumerate(patches):
        p.set_facecolor(cmap(i/len(patches)))
    axes[0].axvline(df['rating'].mean(),color=AMBER,ls='--',lw=1.8,label=f"Mean {df['rating'].mean():.1f}")
    axes[0].set_title('Rating Distribution',color=CREAM,fontsize=12,fontweight='bold',pad=10)
    axes[0].set_xlabel('Rating',color='#aaa',fontsize=10); axes[0].set_ylabel('Count',color='#aaa',fontsize=10)
    axes[0].tick_params(colors='#aaa'); axes[0].legend(facecolor=CARD,edgecolor='#222',labelcolor=CREAM,fontsize=9)
    for sp in axes[0].spines.values(): sp.set_color('#1a1a2a')

    ga  = df.groupby('genre')['rating'].mean().sort_values()
    nrm = (ga-ga.min())/(ga.max()-ga.min())
    cols= [plt.cm.YlOrRd(v) for v in nrm]
    bars= axes[1].barh(ga.index, ga.values, color=cols, edgecolor=BG, linewidth=.5)
    for bar,v in zip(bars,ga.values):
        axes[1].text(v+.02,bar.get_y()+bar.get_height()/2,f'{v:.2f}',va='center',color='#aaa',fontsize=8)
    axes[1].set_title('Avg Rating by Genre',color=CREAM,fontsize=12,fontweight='bold',pad=10)
    axes[1].set_xlabel('Avg Rating',color='#aaa'); axes[1].tick_params(colors='#aaa')
    for sp in axes[1].spines.values(): sp.set_color('#1a1a2a')
    plt.tight_layout(pad=2); st.pyplot(fig); plt.close()

    # Chart 2 — heatmap + scatter
    fig2, axes2 = plt.subplots(1,2,figsize=(13,5))
    fig2.patch.set_facecolor(BG)
    for ax in axes2: ax.set_facecolor(CARD)

    nc   = ['duration_min','budget_million','num_votes','director_score','cast_score','rating']
    corr = df[nc].corr()
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='YlOrRd',ax=axes2[0],
                linewidths=1,linecolor=BG,square=True,cbar_kws={'shrink':.78},
                annot_kws={'size':9,'weight':'bold'})
    axes2[0].set_title('Feature Correlation Heatmap',color=CREAM,fontsize=12,fontweight='bold',pad=10)
    axes2[0].tick_params(colors='#aaa',labelsize=9)
    axes2[0].set_xticklabels(axes2[0].get_xticklabels(),rotation=30,ha='right')

    sc2 = axes2[1].scatter(df['director_score'],df['rating'],
                           alpha=.4,c=df['cast_score'],cmap='YlOrRd',
                           s=24,edgecolors='none')
    cb  = plt.colorbar(sc2,ax=axes2[1]); cb.ax.yaxis.set_tick_params(color='#aaa'); cb.set_label('Cast Score',color='#aaa')
    axes2[1].set_xlabel('Director Score',color='#aaa'); axes2[1].set_ylabel('IMDB Rating',color='#aaa')
    axes2[1].set_title('Director Score vs Rating\n(color = Cast Score)',color=CREAM,fontsize=12,fontweight='bold',pad=10)
    axes2[1].tick_params(colors='#aaa')
    for sp in axes2[1].spines.values(): sp.set_color('#1a1a2a')
    plt.tight_layout(pad=2); st.pyplot(fig2); plt.close()

    st.markdown('<div class="sh">DATASET PREVIEW</div>', unsafe_allow_html=True)
    st.dataframe(
        df[['title','genre','release_year','duration_min','budget_million','director_score','cast_score','rating']].head(10),
        use_container_width=True, hide_index=True
    )

# ──────────────────────────────────────────────────────
# TAB 3 — MODEL RESULTS
# ──────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="sh">MODEL PERFORMANCE</div>', unsafe_allow_html=True)

    mdf = pd.DataFrame({'Model':list(res.keys()),
                        'MAE':  [round(res[m]['MAE'],4)  for m in res],
                        'RMSE': [round(res[m]['RMSE'],4) for m in res],
                        'R² Score':[round(res[m]['R2'],4) for m in res]})
    st.dataframe(
        mdf.style.highlight_max(subset=['R² Score'],color='#1a2d0d')
               .highlight_min(subset=['MAE','RMSE'],color='#1a2d0d')
               .set_properties(**{'font-family':'JetBrains Mono,monospace'}),
        use_container_width=True, hide_index=True
    )

    # Bar charts
    BG='#01090f'; CARD='#040f1a'; CREAM='#EAE2B7'
    fig3, axes3 = plt.subplots(1,3,figsize=(14,4))
    fig3.patch.set_facecolor(BG)
    snames  = ['Lin. Reg.','Rnd Forest','Grad. Boost']
    palette = ['#FCBF49','#F77F00','#D62828']

    for i,(met,ax) in enumerate(zip(['MAE','RMSE','R2'],axes3)):
        vals = [res[m][met] for m in res]
        bars = ax.bar(snames,vals,color=palette,edgecolor=BG,linewidth=1,width=.5)
        ax.set_facecolor(CARD); ax.set_title(met,color=CREAM,fontweight='bold',fontsize=13,pad=10)
        ax.tick_params(colors='#aaa',labelsize=9)
        for sp in ax.spines.values(): sp.set_color('#1a1a2a')
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.004,
                    f'{v:.3f}',ha='center',fontsize=10,color='white',fontweight='bold')
    plt.tight_layout(pad=2); st.pyplot(fig3); plt.close()

    # Actual vs Predicted
    st.markdown('<div class="sh">ACTUAL vs PREDICTED — BEST MODEL</div>', unsafe_allow_html=True)
    cp, cs = st.columns([1,1])

    with cp:
        fig4, ax4 = plt.subplots(figsize=(7,5))
        fig4.patch.set_facecolor(BG); ax4.set_facecolor(CARD)
        pb  = res[best_nm]['preds']
        sc4 = ax4.scatter(yte,pb,alpha=.45,c=pb,cmap='YlOrRd',s=30,edgecolors='none',zorder=3)
        plt.colorbar(sc4,ax=ax4,label='Predicted').ax.yaxis.set_tick_params(color='#aaa')
        lims=[min(yte.min(),pb.min())-.3, max(yte.max(),pb.max())+.3]
        ax4.plot(lims,lims,color='#F77F00',lw=2,ls='--',label='Perfect',zorder=4)
        ax4.set_xlabel('Actual Rating',color='#aaa'); ax4.set_ylabel('Predicted Rating',color='#aaa')
        ax4.set_title(f'Actual vs Predicted\n({best_nm})',color=CREAM,fontweight='bold')
        ax4.tick_params(colors='#aaa')
        ax4.legend(facecolor=CARD,edgecolor='#222',labelcolor=CREAM)
        for sp in ax4.spines.values(): sp.set_color('#1a1a2a')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    with cs:
        st.markdown(f"""
        <div class="rbox" style="animation:none">
          <div style="color:var(--muted);font-size:.73rem;font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase">Best Performing Model</div>
          <div style="font-family:'Bebas Neue',sans-serif;font-size:1.9rem;color:var(--amber);letter-spacing:4px;margin:10px 0">{best_nm}</div>
          <div style="border-top:1px solid rgba(252,191,73,.15);padding-top:16px;">
            <div class="r-mini">
              <div><div class="r-mv">{res[best_nm]['R2']:.3f}</div><div class="r-ml">R² Score</div></div>
              <div><div class="r-mv">{res[best_nm]['MAE']:.3f}</div><div class="r-ml">MAE</div></div>
              <div><div class="r-mv">{res[best_nm]['RMSE']:.3f}</div><div class="r-ml">RMSE</div></div>
            </div>
          </div>
          <div style="margin-top:20px;background:rgba(247,127,0,.07);border-radius:10px;padding:16px;">
            <div style="color:var(--muted);font-size:.72rem;font-family:'JetBrains Mono';letter-spacing:1px;margin-bottom:8px">INTERPRETATION</div>
            <div style="font-size:.95rem;color:var(--cream);line-height:1.65">
              R² = <span style="color:var(--amber)">{res[best_nm]['R2']:.1%}</span> variance explained.<br>
              Predictions within <span style="color:var(--amber)">±{res[best_nm]['MAE']:.2f}</span> rating points on average.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────
st.markdown("""
<div class="ft">
  ◆ &nbsp; Module 5 : AI/ML &nbsp; · &nbsp; IMDB Movie Rating Prediction &nbsp; · &nbsp;
  Scikit-learn &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; Matplotlib &nbsp;·&nbsp; Seaborn &nbsp; ◆
</div>
""", unsafe_allow_html=True)
