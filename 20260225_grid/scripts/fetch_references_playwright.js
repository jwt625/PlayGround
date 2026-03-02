#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const os = require('os');

const SOURCES = [
  ["lbnl_2024_data_center_report", "https://eta.lbl.gov/publications/2024-lbnl-data-center-energy-usage-report"],
  ["doe_release_data_center_demand", "https://www.energy.gov/articles/doe-releases-new-report-evaluating-increase-electricity-demand-data-centers"],
  ["berkeley_queues_landing", "https://emp.lbl.gov/queues"],
  ["osti_queued_up_2025", "https://www.osti.gov/biblio/3008763"],
  ["ferc_order_2023_explainer", "https://www.ferc.gov/explainer-interconnection-final-rule"],
  ["ferc_order_2023a_explainer", "https://www.ferc.gov/explainer-interconnection-final-rule-2023-A"],
  ["pjm_2026_2027_bra", "https://insidelines.pjm.com/pjm-auction-procures-134311-mw-of-generation-resources-supply-responds-to-price-signal/"],
  ["pjm_2027_2028_bra", "https://insidelines.pjm.com/pjm-auction-procures-134479-mw-of-generation-resources/"],
  ["wecc_large_load_report", "https://www.wecc.org/wecc-document/19111"],
  ["ferc_nerc_ltra_2025_presentation", "https://www.ferc.gov/news-events/news/ferc-nerc-presentation-2025-long-term-reliability-assessment"],
  ["eia_steo", "https://www.eia.gov/steo"],
  ["eia_electricity_monthly", "https://www.eia.gov/electricity/monthly/index.php"],
  ["eia_aeo2025_lcoe_pdf", "https://www.eia.gov/outlooks/aeo/electricity_generation/pdf/AEO2025_LCOE_report.pdf"],
  ["treasury_45y_48e_release", "https://home.treasury.gov/news/press-releases/jy2787"],
  ["bloom_equinix_100mw", "https://investor.bloomenergy.com/press-releases/press-release-details/2025/Bloom-Energy-Expands-Data-Center-Power-Agreement-with-Equinix-Surpassing-100MW/default.aspx"],
  ["openai_stargate", "https://openai.com/blog/announcing-the-stargate-project/"],
];

const RAW_DIR = path.join('references', 'raw');
const MD_DIR = path.join('references', 'md');
const MANIFEST = path.join('references', 'manifest.playwright.json');

function normalizeText(t) {
  return t.replace(/\u00a0/g, ' ').replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

async function run() {
  fs.mkdirSync(RAW_DIR, { recursive: true });
  fs.mkdirSync(MD_DIR, { recursive: true });

  const { firefox } = require('playwright');

  const userDataDir = process.env.PW_USER_DATA_DIR || path.join('references', '.pw-firefox-profile');
  fs.mkdirSync(userDataDir, { recursive: true });
  const context = await firefox.launchPersistentContext(userDataDir, {
    headless: true,
    viewport: { width: 1440, height: 1800 },
    ignoreHTTPSErrors: true,
  });

  const queue = [...SOURCES];
  const results = [];
  const workers = 4;

  async function worker(idx) {
    while (queue.length) {
      const item = queue.shift();
      if (!item) break;
      const [key, url] = item;
      const retrieved = new Date().toISOString();
      const page = await context.newPage();
      try {
        if (url.toLowerCase().endsWith('.pdf')) {
          const outPdf = path.join(RAW_DIR, `${key}.pdf`);
          const pdfResp = await context.request.get(url);
          const buff = await pdfResp.body();
          fs.writeFileSync(outPdf, buff);
          const status = pdfResp.status();
          const md = `---\nkey: ${key}\nurl: ${url}\nfinal_url: ${url}\nretrieved_at_utc: ${retrieved}\nsource_type: pdf\nsource_method: playwright_firefox\nhttp_status: ${status}\nraw_path: ${outPdf}\n---\n\nPDF downloaded. Text extraction is handled by the Python pipeline.\n`;
          const mdPath = path.join(MD_DIR, `${key}.md`);
          fs.writeFileSync(mdPath, md, 'utf8');
          console.log(`[ok] ${key} (pdf) by subagent-${idx}`);
          results.push({ key, url, status: 'ok', http_status: status, source_type: 'pdf', raw_path: outPdf, md_path: mdPath, error: null });
          continue;
        }

        const resp = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
        await page.waitForTimeout(1500);
        const finalUrl = page.url();
        const ctype = (resp && resp.headers()['content-type']) || '';
        const status = resp ? resp.status() : null;

        if (ctype.toLowerCase().includes('application/pdf') || finalUrl.toLowerCase().endsWith('.pdf')) {
          const outPdf = path.join(RAW_DIR, `${key}.pdf`);
          const pdfResp = await page.request.get(finalUrl);
          const buff = await pdfResp.body();
          fs.writeFileSync(outPdf, buff);
          const md = `---\nkey: ${key}\nurl: ${url}\nfinal_url: ${finalUrl}\nretrieved_at_utc: ${retrieved}\nsource_type: pdf\nsource_method: playwright_firefox\nhttp_status: ${status}\nraw_path: ${outPdf}\n---\n\nPDF downloaded. Text extraction is handled by the Python pipeline.\n`;
          const mdPath = path.join(MD_DIR, `${key}.md`);
          fs.writeFileSync(mdPath, md, 'utf8');
          console.log(`[ok] ${key} (pdf) by subagent-${idx}`);
          results.push({ key, url, status: 'ok', http_status: status, source_type: 'pdf', raw_path: outPdf, md_path: mdPath, error: null });
        } else {
          const html = await page.content();
          const rawHtml = path.join(RAW_DIR, `${key}.html`);
          fs.writeFileSync(rawHtml, html, 'utf8');

          const extracted = await page.evaluate(() => {
            const pick = document.querySelector('article') || document.querySelector('main') || document.body;
            return {
              title: document.title || '',
              byline: (document.querySelector('[rel="author"], .author, .byline') || {}).textContent || '',
              text: (pick && pick.innerText) ? pick.innerText : (document.body ? document.body.innerText : ''),
              selector: pick === document.body ? 'body' : (pick.tagName || 'body').toLowerCase(),
            };
          });

          const text = normalizeText(`${extracted.title}\n\n${extracted.byline || ''}\n\n${extracted.text || ''}`);
          const mdPath = path.join(MD_DIR, `${key}.md`);
          const md = [
            '---',
            `key: ${key}`,
            `url: ${url}`,
            `final_url: ${finalUrl}`,
            `retrieved_at_utc: ${retrieved}`,
            `source_type: html`,
            `source_method: playwright_firefox`,
            `extracted_selector: ${extracted.selector}`,
            `http_status: ${status}`,
            `raw_path: ${rawHtml}`,
            '---',
            '',
            text || 'No text extracted.',
            '',
          ].join('\n');
          fs.writeFileSync(mdPath, md, 'utf8');
          console.log(`[ok] ${key} (html) by subagent-${idx}`);
          results.push({ key, url, status: 'ok', http_status: status, source_type: 'html', raw_path: rawHtml, md_path: mdPath, error: null });
        }
      } catch (e) {
        const mdPath = path.join(MD_DIR, `${key}.md`);
        if (!fs.existsSync(mdPath)) {
          fs.writeFileSync(mdPath, `---\nkey: ${key}\nurl: ${url}\nretrieved_at_utc: ${retrieved}\nstatus: failed\nerror: ${String(e).replace(/\n/g, ' ')}\n---\n`, 'utf8');
        }
        console.log(`[failed] ${key} by subagent-${idx}: ${e.message}`);
        results.push({ key, url, status: 'failed', http_status: null, source_type: 'unknown', raw_path: null, md_path: mdPath, error: String(e) });
      } finally {
        await page.close();
      }
    }
  }

  await Promise.all(Array.from({ length: workers }, (_, i) => worker(i + 1)));
  await context.close();

  results.sort((a, b) => a.key.localeCompare(b.key));
  const out = {
    generated_at_utc: new Date().toISOString(),
    total: results.length,
    ok: results.filter(r => r.status === 'ok').length,
    failed: results.filter(r => r.status !== 'ok').length,
    user_data_dir: userDataDir,
    items: results,
  };
  fs.writeFileSync(MANIFEST, JSON.stringify(out, null, 2), 'utf8');
  console.log(`Manifest written: ${MANIFEST}`);
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
