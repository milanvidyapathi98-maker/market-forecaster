"""Built-in list of assets for each market, shown in the dropdowns.

Format:  "Group name": [(Yahoo Finance symbol, display name), ...]
To add an asset, add a line in the right group. To check a symbol, search for it
on finance.yahoo.com and copy the symbol shown in brackets.
"""

CATALOG = {
    # ------------------------------------------------------------------ SHARES
    "shares": {
        "US · Large caps": [
            ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
            ("AMZN", "Amazon"), ("GOOGL", "Alphabet (Class A)"), ("GOOG", "Alphabet (Class C)"),
            ("META", "Meta Platforms"), ("TSLA", "Tesla"), ("AVGO", "Broadcom"),
            ("BRK-B", "Berkshire Hathaway"), ("LLY", "Eli Lilly"), ("JPM", "JPMorgan Chase"),
            ("V", "Visa"), ("MA", "Mastercard"), ("UNH", "UnitedHealth"), ("XOM", "Exxon Mobil"),
            ("WMT", "Walmart"), ("JNJ", "Johnson & Johnson"), ("PG", "Procter & Gamble"),
            ("HD", "Home Depot"), ("COST", "Costco"), ("ORCL", "Oracle"), ("NFLX", "Netflix"),
            ("ABBV", "AbbVie"), ("BAC", "Bank of America"), ("KO", "Coca-Cola"),
            ("CRM", "Salesforce"), ("CVX", "Chevron"), ("MRK", "Merck & Co"), ("AMD", "AMD"),
            ("PEP", "PepsiCo"), ("ADBE", "Adobe"), ("TMO", "Thermo Fisher"), ("LIN", "Linde"),
            ("ACN", "Accenture"), ("MCD", "McDonald's"), ("CSCO", "Cisco"), ("ABT", "Abbott"),
            ("WFC", "Wells Fargo"), ("IBM", "IBM"), ("GE", "GE Aerospace"), ("PM", "Philip Morris"),
            ("INTU", "Intuit"), ("NOW", "ServiceNow"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"),
            ("DIS", "Walt Disney"), ("CAT", "Caterpillar"), ("ISRG", "Intuitive Surgical"),
            ("AMGN", "Amgen"), ("VZ", "Verizon"), ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"),
            ("PFE", "Pfizer"), ("RTX", "RTX"), ("UBER", "Uber"), ("T", "AT&T"), ("SPGI", "S&P Global"),
            ("BKNG", "Booking Holdings"), ("AXP", "American Express"), ("LOW", "Lowe's"),
            ("HON", "Honeywell"), ("UNP", "Union Pacific"), ("BLK", "BlackRock"), ("DHR", "Danaher"),
            ("NEE", "NextEra Energy"), ("PLTR", "Palantir"), ("C", "Citigroup"), ("SCHW", "Charles Schwab"),
            ("DE", "Deere"), ("BA", "Boeing"), ("LMT", "Lockheed Martin"), ("SBUX", "Starbucks"),
            ("GILD", "Gilead Sciences"), ("BMY", "Bristol-Myers Squibb"), ("MDT", "Medtronic"),
            ("INTC", "Intel"), ("MU", "Micron"), ("AMAT", "Applied Materials"), ("LRCX", "Lam Research"),
            ("KLAC", "KLA"), ("PANW", "Palo Alto Networks"), ("CRWD", "CrowdStrike"),
            ("ANET", "Arista Networks"), ("TMUS", "T-Mobile US"), ("CMCSA", "Comcast"),
            ("NKE", "Nike"), ("MO", "Altria"), ("MDLZ", "Mondelez"), ("CL", "Colgate-Palmolive"),
            ("UPS", "UPS"), ("FDX", "FedEx"), ("GM", "General Motors"), ("F", "Ford"),
            ("PYPL", "PayPal"), ("COF", "Capital One"), ("USB", "U.S. Bancorp"), ("MET", "MetLife"),
            ("AIG", "AIG"), ("CVS", "CVS Health"), ("TGT", "Target"), ("DUK", "Duke Energy"),
            ("SO", "Southern Company"), ("AMT", "American Tower"), ("SPG", "Simon Property"),
            ("MMM", "3M"), ("EMR", "Emerson Electric"), ("GD", "General Dynamics"), ("COP", "ConocoPhillips"),
            ("ABNB", "Airbnb"), ("SHOP", "Shopify"), ("SPOT", "Spotify"), ("COIN", "Coinbase"),
            ("ARM", "Arm Holdings"), ("TSM", "TSMC (US listing)"), ("ASML", "ASML (US listing)"),
            ("NVO", "Novo Nordisk (US listing)"),
        ],
        "Germany · DAX": [
            ("SAP.DE", "SAP"), ("SIE.DE", "Siemens"), ("ALV.DE", "Allianz"),
            ("DTE.DE", "Deutsche Telekom"), ("AIR.DE", "Airbus"), ("MUV2.DE", "Munich Re"),
            ("RHM.DE", "Rheinmetall"), ("ENR.DE", "Siemens Energy"), ("MBG.DE", "Mercedes-Benz"),
            ("BMW.DE", "BMW"), ("VOW3.DE", "Volkswagen (Pref.)"), ("DBK.DE", "Deutsche Bank"),
            ("DB1.DE", "Deutsche Börse"), ("IFX.DE", "Infineon"), ("BAS.DE", "BASF"),
            ("BAYN.DE", "Bayer"), ("ADS.DE", "Adidas"), ("DHL.DE", "DHL Group"),
            ("EOAN.DE", "E.ON"), ("RWE.DE", "RWE"), ("CBK.DE", "Commerzbank"),
            ("HNR1.DE", "Hannover Rück"), ("MTX.DE", "MTU Aero Engines"), ("SHL.DE", "Siemens Healthineers"),
            ("DTG.DE", "Daimler Truck"), ("HEI.DE", "Heidelberg Materials"), ("HEN3.DE", "Henkel (Pref.)"),
            ("BEI.DE", "Beiersdorf"), ("MRK.DE", "Merck KGaA"), ("FRE.DE", "Fresenius"),
            ("FME.DE", "Fresenius Medical Care"), ("VNA.DE", "Vonovia"), ("CON.DE", "Continental"),
            ("PAH3.DE", "Porsche SE"), ("P911.DE", "Porsche AG"), ("SY1.DE", "Symrise"),
            ("BNR.DE", "Brenntag"), ("G1A.DE", "GEA Group"), ("ZAL.DE", "Zalando"),
            ("SRT3.DE", "Sartorius (Pref.)"), ("QIA.DE", "Qiagen"),
        ],
        "Europe · Major shares": [
            ("ASML.AS", "ASML"), ("MC.PA", "LVMH"), ("OR.PA", "L'Oréal"), ("RMS.PA", "Hermès"),
            ("TTE.PA", "TotalEnergies"), ("SAN.PA", "Sanofi"), ("AI.PA", "Air Liquide"),
            ("SU.PA", "Schneider Electric"), ("BNP.PA", "BNP Paribas"), ("NESN.SW", "Nestlé"),
            ("NOVN.SW", "Novartis"), ("ROG.SW", "Roche"), ("UBSG.SW", "UBS"),
            ("NOVO-B.CO", "Novo Nordisk"), ("SHEL.L", "Shell"), ("AZN.L", "AstraZeneca"),
            ("HSBA.L", "HSBC"), ("ULVR.L", "Unilever"), ("BP.L", "BP"), ("RIO.L", "Rio Tinto"),
            ("SAN.MC", "Banco Santander"), ("ITX.MC", "Inditex"), ("ISP.MI", "Intesa Sanpaolo"),
            ("ENEL.MI", "Enel"),
        ],
        "Asia · Major shares": [
            ("7203.T", "Toyota"), ("6758.T", "Sony"), ("9984.T", "SoftBank Group"),
            ("8306.T", "Mitsubishi UFJ"), ("2330.TW", "TSMC"), ("005930.KS", "Samsung Electronics"),
            ("0700.HK", "Tencent"), ("9988.HK", "Alibaba"), ("1299.HK", "AIA Group"),
            ("RELIANCE.NS", "Reliance Industries"), ("TCS.NS", "Tata Consultancy Services"),
            ("INFY.NS", "Infosys"), ("HDFCBANK.NS", "HDFC Bank"),
        ],
        "Indices": [
            ("^GSPC", "S&P 500"), ("^NDX", "Nasdaq 100"), ("^DJI", "Dow Jones"),
            ("^RUT", "Russell 2000"), ("^GDAXI", "DAX"), ("^STOXX50E", "Euro Stoxx 50"),
            ("^FTSE", "FTSE 100"), ("^FCHI", "CAC 40"), ("^N225", "Nikkei 225"),
            ("^HSI", "Hang Seng"), ("^NSEI", "Nifty 50"), ("^VIX", "VIX volatility index"),
        ],
        "ETFs": [
            ("SPY", "SPDR S&P 500 ETF"), ("VOO", "Vanguard S&P 500 ETF"), ("QQQ", "Invesco QQQ (Nasdaq 100)"),
            ("DIA", "SPDR Dow Jones ETF"), ("IWM", "iShares Russell 2000"), ("VTI", "Vanguard Total US Market"),
            ("VT", "Vanguard Total World"), ("EFA", "iShares MSCI EAFE"), ("EEM", "iShares MSCI Emerging Markets"),
            ("VGK", "Vanguard Europe"), ("EXS1.DE", "iShares Core DAX (Xetra)"),
            ("EUNL.DE", "iShares Core MSCI World (Xetra)"), ("XLK", "Technology Select Sector"),
            ("XLF", "Financial Select Sector"), ("XLE", "Energy Select Sector"),
            ("XLV", "Health Care Select Sector"), ("SMH", "VanEck Semiconductor ETF"),
            ("ARKK", "ARK Innovation ETF"),
        ],
    },

    # ------------------------------------------------------------------- FOREX
    "forex": {
        "Majors": [
            ("EURUSD=X", "Euro / US Dollar"), ("GBPUSD=X", "British Pound / US Dollar"),
            ("USDJPY=X", "US Dollar / Japanese Yen"), ("USDCHF=X", "US Dollar / Swiss Franc"),
            ("AUDUSD=X", "Australian Dollar / US Dollar"), ("USDCAD=X", "US Dollar / Canadian Dollar"),
            ("NZDUSD=X", "New Zealand Dollar / US Dollar"),
        ],
        "Euro crosses": [
            ("EURGBP=X", "Euro / British Pound"), ("EURJPY=X", "Euro / Japanese Yen"),
            ("EURCHF=X", "Euro / Swiss Franc"), ("EURAUD=X", "Euro / Australian Dollar"),
            ("EURCAD=X", "Euro / Canadian Dollar"), ("EURNZD=X", "Euro / New Zealand Dollar"),
            ("EURSEK=X", "Euro / Swedish Krona"), ("EURNOK=X", "Euro / Norwegian Krone"),
            ("EURDKK=X", "Euro / Danish Krone"), ("EURPLN=X", "Euro / Polish Zloty"),
            ("EURCZK=X", "Euro / Czech Koruna"), ("EURHUF=X", "Euro / Hungarian Forint"),
            ("EURTRY=X", "Euro / Turkish Lira"), ("EURINR=X", "Euro / Indian Rupee"),
            ("EURCNY=X", "Euro / Chinese Yuan"),
        ],
        "Other crosses": [
            ("GBPJPY=X", "British Pound / Japanese Yen"), ("GBPCHF=X", "British Pound / Swiss Franc"),
            ("GBPAUD=X", "British Pound / Australian Dollar"), ("GBPCAD=X", "British Pound / Canadian Dollar"),
            ("AUDJPY=X", "Australian Dollar / Japanese Yen"), ("AUDNZD=X", "Australian Dollar / NZ Dollar"),
            ("AUDCAD=X", "Australian Dollar / Canadian Dollar"), ("CADJPY=X", "Canadian Dollar / Japanese Yen"),
            ("CHFJPY=X", "Swiss Franc / Japanese Yen"), ("NZDJPY=X", "NZ Dollar / Japanese Yen"),
        ],
        "US Dollar · Emerging": [
            ("USDCNY=X", "US Dollar / Chinese Yuan"), ("USDHKD=X", "US Dollar / Hong Kong Dollar"),
            ("USDSGD=X", "US Dollar / Singapore Dollar"), ("USDINR=X", "US Dollar / Indian Rupee"),
            ("USDLKR=X", "US Dollar / Sri Lankan Rupee"), ("USDKRW=X", "US Dollar / South Korean Won"),
            ("USDTHB=X", "US Dollar / Thai Baht"), ("USDMXN=X", "US Dollar / Mexican Peso"),
            ("USDBRL=X", "US Dollar / Brazilian Real"), ("USDZAR=X", "US Dollar / South African Rand"),
            ("USDTRY=X", "US Dollar / Turkish Lira"), ("USDPLN=X", "US Dollar / Polish Zloty"),
        ],
    },

    # ------------------------------------------------------------------ CRYPTO
    "crypto": {
        "Top coins": [
            ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("XRP-USD", "XRP"),
            ("BNB-USD", "BNB"), ("SOL-USD", "Solana"), ("DOGE-USD", "Dogecoin"),
            ("ADA-USD", "Cardano"), ("TRX-USD", "TRON"), ("LINK-USD", "Chainlink"),
            ("AVAX-USD", "Avalanche"), ("XLM-USD", "Stellar"), ("DOT-USD", "Polkadot"),
            ("LTC-USD", "Litecoin"), ("BCH-USD", "Bitcoin Cash"), ("HBAR-USD", "Hedera"),
            ("SUI20947-USD", "Sui"), ("TON11419-USD", "Toncoin"),
        ],
        "More coins": [
            ("SHIB-USD", "Shiba Inu"), ("PEPE24478-USD", "Pepe"), ("UNI7083-USD", "Uniswap"),
            ("AAVE-USD", "Aave"), ("NEAR-USD", "NEAR Protocol"), ("ATOM-USD", "Cosmos"),
            ("ETC-USD", "Ethereum Classic"), ("XMR-USD", "Monero"), ("FIL-USD", "Filecoin"),
            ("ICP-USD", "Internet Computer"), ("ALGO-USD", "Algorand"), ("APT21794-USD", "Aptos"),
            ("ARB11841-USD", "Arbitrum"), ("VET-USD", "VeChain"), ("XTZ-USD", "Tezos"),
        ],
        "In euros": [
            ("BTC-EUR", "Bitcoin in EUR"), ("ETH-EUR", "Ethereum in EUR"),
            ("SOL-EUR", "Solana in EUR"), ("XRP-EUR", "XRP in EUR"),
        ],
    },

    # -------------------------------------------------------------- TREASURIES
    "treasuries": {
        "US Treasury yields": [
            ("^IRX", "13-week T-bill yield"), ("^FVX", "5-year Treasury yield"),
            ("^TNX", "10-year Treasury yield"), ("^TYX", "30-year Treasury yield"),
        ],
        "US Treasury ETFs": [
            ("BIL", "1-3 month T-bills (BIL)"), ("SGOV", "0-3 month T-bills (SGOV)"),
            ("SHY", "1-3 year Treasuries (SHY)"), ("VGSH", "Short-term Treasuries (VGSH)"),
            ("IEI", "3-7 year Treasuries (IEI)"), ("IEF", "7-10 year Treasuries (IEF)"),
            ("VGIT", "Intermediate Treasuries (VGIT)"), ("TLH", "10-20 year Treasuries (TLH)"),
            ("TLT", "20+ year Treasuries (TLT)"), ("VGLT", "Long-term Treasuries (VGLT)"),
            ("GOVT", "All US Treasuries (GOVT)"), ("TIP", "Inflation-protected TIPS (TIP)"),
        ],
        "Other bond ETFs": [
            ("AGG", "US aggregate bonds (AGG)"), ("BND", "Total US bond market (BND)"),
            ("LQD", "Investment-grade corporate (LQD)"), ("HYG", "High-yield corporate (HYG)"),
            ("EMB", "Emerging-market bonds (EMB)"), ("BNDX", "International bonds (BNDX)"),
            ("EUNH.DE", "Euro government bonds (Xetra)"),
            ("IS0L.DE", "German government bonds (Xetra)"),
        ],
    },

    # ------------------------------------------------------------------- METALS
    "gold": {
        "Metals (futures)": [
            ("GC=F", "Gold"), ("SI=F", "Silver"), ("PL=F", "Platinum"),
            ("PA=F", "Palladium"), ("HG=F", "Copper"), ("ALI=F", "Aluminium"),
        ],
        "Metal ETFs": [
            ("GLD", "SPDR Gold Shares (GLD)"), ("IAU", "iShares Gold Trust (IAU)"),
            ("SGOL", "abrdn Physical Gold (SGOL)"), ("4GLD.DE", "Xetra-Gold, in EUR"),
            ("SLV", "iShares Silver Trust (SLV)"), ("PPLT", "abrdn Platinum (PPLT)"),
            ("PALL", "abrdn Palladium (PALL)"), ("CPER", "US Copper Index Fund (CPER)"),
        ],
        "Mining companies": [
            ("GDX", "Gold miners ETF (GDX)"), ("GDXJ", "Junior gold miners ETF (GDXJ)"),
            ("SIL", "Silver miners ETF (SIL)"), ("NEM", "Newmont"), ("AEM", "Agnico Eagle"),
            ("B", "Barrick Mining"), ("WPM", "Wheaton Precious Metals"), ("FCX", "Freeport-McMoRan"),
        ],
    },
}

# Quick lookups: symbol -> name, and symbol -> group
NAME_OF = {sym: name for groups in CATALOG.values()
           for items in groups.values() for sym, name in items}
GROUP_OF = {sym: group for groups in CATALOG.values()
            for group, items in groups.items() for sym, _ in items}


def symbols(asset: str, group: str = None) -> list:
    """All catalogue symbols for a market, or for one group of it."""
    groups = CATALOG[asset]
    if group:
        return [s for s, _ in groups.get(group, [])]
    return [s for items in groups.values() for s, _ in items]
