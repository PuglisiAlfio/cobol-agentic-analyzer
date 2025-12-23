from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP(
    name="Accounting System MCP Server"
)

# Percorsi dei file del progetto
PROJECT_DIR = r"MYPATH"
ACCOUNTING_FILE = os.path.join(PROJECT_DIR, "ACCOUNTING_SYSTEM.COB")
BUYROUTINE_FILE = os.path.join(PROJECT_DIR, "BUYROUTINE.COB")
DATABASE_FILE = os.path.join(PROJECT_DIR, "DATABASE.txt")
PRODUCTS_FILE = os.path.join(PROJECT_DIR, "products.txt")

# -------------------------
# Risorse MCP
# -------------------------

@mcp.resource("file://ACCOUNTING_SYSTEM")
def get_accounting_system() -> str:
    """Restituisce il contenuto del file ACCOUNTING_SYSTEM.COB"""
    try:
        with open(ACCOUNTING_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Errore durante la lettura del file ACCOUNTING_SYSTEM.COB: {str(e)}"

@mcp.resource("file://BUYROUTINE")
def get_buyroutine() -> str:
    """Restituisce il contenuto del file BUYROUTINE.COB"""
    try:
        with open(BUYROUTINE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Errore durante la lettura del file BUYROUTINE.COB: {str(e)}"

@mcp.resource("file://DATABASE")
def get_database() -> str:
    """Restituisce il contenuto del file DATABASE.txt"""
    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Errore durante la lettura del file DATABASE.txt: {str(e)}"

@mcp.resource("file://PRODUCTS")
def get_products() -> str:
    """Restituisce il contenuto del file products.txt"""
    try:
        with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Errore durante la lettura del file products.txt: {str(e)}"

# -------------------------
# Avvio server
# -------------------------
if __name__ == '__main__':

    mcp.run(transport="stdio")
