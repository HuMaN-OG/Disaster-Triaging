/* Browser smoke test with explicit, synthetic API fixtures.
   This proves frontend workflows, NOT live ML/production-backend correctness. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const baseIncident = {
  id:42, title:'Flood near bridge', description:'Water has risen near the pedestrian bridge.',
  disaster_type:'Flood', priority_score:87.6, priority_level:'Critical', confidence:0.93,
  status:'Pending', language_detected:'English', image_url:null, created_at:'2026-11-19T12:00:00',
  updated_at:'2026-11-19T12:00:00',
  location:{latitude:28.61,longitude:77.20,address_text:'Example bridge'},
  class_probabilities:{Flood:.93,Earthquake:.03,'Urban Fire':.04},
  recommendations:['Confirm location with local experts','Review suitable resources'],
  status_history:[], attachments:[]
};
const stats={total_incidents:1,critical:1,high:0,medium:0,low:0,very_low:0,
  pending:1,assigned:0,en_route:0,in_progress:0,completed:0};
const analytics={disaster_distribution:{Flood:1},priority_distribution:{Critical:1},
  status_distribution:{Pending:1},language_distribution:{English:1},
  daily_incidents:{'2026-11-19':1},average_confidence:.93};
const brief={
 incident_id:42,disaster_type:'Flood',priority_score:87.6,priority_level:'Critical',
 confidence:.93,citizen_report:baseIncident.description,detected_language:'English',
 location:'Example bridge',latitude:28.61,longitude:77.2,
 created_at:baseIncident.created_at,current_status:'Pending',
 assigned_responder:null,assigned_team:null,timeline:[],attachments:[],
 risk_summary:'This simulated test report warrants review by trained people.',
 immediate_actions:['Validate the report'],recommended_resources:['Example resource']
};
async function main() {
  fs.mkdirSync('artifacts',{recursive:true});
  const browser=await chromium.launch({headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:980}});
  const errors=[];
  page.on('pageerror',e=>{ errors.push(e.message); console.error('PAGE ERROR:',e.message); });
  page.on('console',m=>{ if(m.type()==='error') console.error('BROWSER CONSOLE:',m.text()); });
  page.on('response',r=>{if(r.status()>=400) console.error('HTTP FAILURE:',r.status(),r.url());});
  page.on('requestfailed',r=>console.error('REQUEST FAILURE:',r.url(),r.failure()?.errorText));
  let incident={...baseIncident};
  await page.route('**/api/**',async route=>{
    const request=route.request(),u=new URL(request.url()),p=u.pathname,m=request.method();
    let data=null,status=200;
    if(p==='/api/health')data={status:'ok',database:'connected',models_loaded:true};
    else if(p==='/api/auth/me')data={id:2,email:'operator@example.com',full_name:'Test Operator',role:'Responder',is_active:true,created_at:'2026-11-19T12:00:00'};
    else if(p==='/api/auth/login' && m==='POST')data={access_token:'synthetic-fixture-token',token_type:'bearer',role:'Responder',full_name:'Test Operator',user_id:2};
    else if(p==='/api/incidents' && m==='POST')data={...incident,id:43,title:'Citizen test submission',status:'Pending',description:'Flood water is rising near my house, help is needed'};
    else if(p==='/api/incidents' && m==='GET')data=[incident];
    else if(p==='/api/incidents/42/brief' && m==='GET')data={...brief,current_status:incident.status};
    else if(p==='/api/incidents/42' && m==='GET')data=incident;
    else if(p==='/api/responder/42/accept' && m==='POST'){
      incident={...incident,status:'Assigned',assigned_responder_id:2,
        status_history:[{id:1,new_status:'Assigned',notes:'Accepted by test responder',created_at:'2026-11-19T12:05:00'}]};
      data=incident;
    }
    else if(p==='/api/dashboard/stats')data=stats;
    else if(p==='/api/dashboard/map')data=[{id:42,title:'Flood near bridge',latitude:28.61,longitude:77.2,priority_score:87.6,priority_level:'Critical',status:incident.status,disaster_type:'Flood'}];
    else if(p==='/api/dashboard/analytics')data=analytics;
    else{status=404;data={detail:'Fixture does not implement '+m+' '+p};}
    await route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
  });
  try{
    await page.goto('http://127.0.0.1:4173/',{waitUntil:'domcontentloaded'});
    await page.getByRole('heading',{name:/Every report matters/}).waitFor({timeout:15000});
    await page.getByText('API connected').waitFor();
    await page.screenshot({path:'artifacts/disastersignal-report.png',fullPage:true});

    // Citizen reporting: no implicit location, then a successful synthetic POST.
    assert.equal(await page.getByRole('button',{name:/Submit for triage/}).isDisabled(),true);
    await page.locator('#incident-description').fill('Flood waters are rising and local residents need assistance.');
    await page.locator('#latitude').fill('28.61');
    await page.locator('#longitude').fill('77.20');
    await page.getByRole('button',{name:/Submit for triage/}).click();
    await page.getByRole('heading',{name:'Triage analysis ready'}).waitFor();
    assert.ok((await page.locator('#triage-results').innerText()).includes('No responders have been automatically dispatched.'));

    // Backend-driven queue, map markers, incident drawer and response brief.
    await page.getByRole('button',{name:/Response command/}).click();
    await page.getByRole('heading',{name:/Response command/}).waitFor();
    await page.getByRole('button',{name:'View report 42'}).waitFor();
    await page.evaluate(() => {document.documentElement.style.scrollBehavior='auto'; window.scrollTo(0,0);});
    await page.waitForTimeout(220);
    await page.screenshot({path:'artifacts/disastersignal-command.png',fullPage:true});
    await page.getByRole('button',{name:'View report 42'}).click();
    await page.getByRole('dialog',{name:/Flood near bridge/}).waitFor();
    await page.getByRole('button',{name:/Open brief/}).click();
    await page.getByRole('dialog',{name:'Response brief'}).waitFor();
    assert.ok((await page.getByRole('dialog',{name:'Response brief'}).innerText()).toLowerCase().includes('derived'));
    await page.getByRole('button',{name:'Close response brief'}).click();
    await page.getByRole('button',{name:'Close incident details'}).click();

    // Auth through real UI, then a mocked authorized accept transition.
    await page.getByRole('button',{name:'Sign in'}).click();
    await page.locator('#auth-email').fill('operator@example.com');
    await page.locator('#auth-password').fill('example-password');
    await page.getByRole('button',{name:'Sign in to workspace'}).click();
    await page.getByRole('button',{name:'View report 42'}).click();
    await page.getByRole('button',{name:/Accept incident/}).waitFor();
    await page.getByRole('button',{name:/Accept incident/}).click();
    await page.locator('.drawer-pills .status-pill').getByText('Assigned', { exact: true }).waitFor({ timeout: 10000 });
    await page.getByRole('button',{name:'Close incident details'}).click();

    await page.getByRole('button',{name:/Intelligence hub/}).click();
    await page.getByRole('heading',{name:/Intelligence hub/}).waitFor();
    await page.getByText('Incidents over time').waitFor();
    await page.evaluate(() => {document.documentElement.style.scrollBehavior='auto'; window.scrollTo(0,0);});
    await page.waitForTimeout(220);
    await page.screenshot({path:'artifacts/disastersignal-analytics.png',fullPage:true});

    // Regression: the active Leaflet map must remain BELOW the mobile drawer.
    await page.locator('.side-link').filter({hasText:'Report an incident'}).click();
    await page.getByRole('button',{name:/Choose on map/}).click();
    await page.setViewportSize({width:390,height:844});
    await page.getByRole('button',{name:'Open navigation'}).click();
    assert.equal(await page.locator('.dt-sidebar').evaluate(x=>x.classList.contains('is-open')),true);
    await page.waitForTimeout(300);
    const topAtMap = await page.evaluate(() => {
      const top = document.elementFromPoint(100, Math.min(640, window.innerHeight - 100));
      return !!top?.closest('.dt-sidebar');
    });
    assert.ok(topAtMap, 'Leaflet map panes must not overlay the mobile navigation drawer');
    await page.screenshot({path:'artifacts/disastersignal-mobile.png',fullPage:true});
    assert.deepEqual(errors,[],'Browser runtime errors');
    console.log('DisasterSignal synthetic browser smoke PASSED: reporting, queue, GIS, brief, responder sign-in/actions, analytics and mobile navigation.');
  } catch(err) {
    console.error('BROWSER DEBUG TITLE:',await page.title());
    console.error('BROWSER DEBUG BODY:',(await page.locator('body').innerText().catch(()=>'' )).slice(0,2400));
    try { await page.screenshot({path:'artifacts/ui-failure.png',fullPage:true}); } catch {}
    console.error('PAGE ERRORS:',errors);
    throw err;
  } finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
