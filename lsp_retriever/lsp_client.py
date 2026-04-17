import subprocess
import json
import threading
import time
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote

class LSPClient:
    def __init__(self, command, root_uri):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        self.root_uri = root_uri
        self.id_counter = 1
        self.responses = {}
        self.lock = threading.Lock()
        
        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        
        # Start stderr reader
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.stderr_thread.start()

    def _stderr_loop(self):
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            # Silent by default, can be enabled for debugging
            # print(f"LSP LOG: {line.decode('utf-8').strip()}")

    def _read_loop(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line_decoded = line.decode('ascii').strip()
            if line_decoded.startswith("Content-Length: "):
                length = int(line_decoded.split(": ")[1])
                # Skip the empty line
                while line_decoded:
                    line_decoded = self.process.stdout.readline().decode('ascii').strip()
                
                content = self.process.stdout.read(length)
                try:
                    message = json.loads(content.decode('utf-8'))
                    if 'id' in message:
                        with self.lock:
                            self.responses[message['id']] = message
                    else:
                        # Handle notifications if necessary
                        pass
                except Exception as e:
                    logging.error(f"Error decoding LSP JSON: {e}")

    def send_request(self, method, params):
        with self.lock:
            req_id = self.id_counter
            self.id_counter += 1
        
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        }
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        self.process.stdin.write(message.encode('utf-8'))
        self.process.stdin.flush()
        return req_id

    def send_notification(self, method, params):
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"
        self.process.stdin.write(message.encode('utf-8'))
        self.process.stdin.flush()

    def wait_for_response(self, req_id, timeout=15):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if req_id in self.responses:
                    return self.responses.pop(req_id)
            time.sleep(0.05)
        return None

    def initialize(self):
        params = {
            "processId": None,
            "rootUri": self.root_uri,
            "capabilities": {
                "workspace": {
                    "symbol": {
                        "dynamicRegistration": True,
                        "symbolKind": {"valueSet": [k for k in range(1, 27)]}
                    }
                },
                "textDocument": {
                    "definition": {"dynamicRegistration": True},
                    "references": {"dynamicRegistration": True},
                    "documentSymbol": {"dynamicRegistration": True}
                }
            }
        }
        req_id = self.send_request("initialize", params)
        res = self.wait_for_response(req_id)
        self.send_notification("initialized", {})
        return res

    def workspace_symbol(self, query):
        params = {"query": query}
        req_id = self.send_request("workspace/symbol", params)
        return self.wait_for_response(req_id)

    def document_symbol(self, uri):
        params = {"textDocument": {"uri": uri}}
        req_id = self.send_request("textDocument/documentSymbol", params)
        return self.wait_for_response(req_id)

    def definition(self, uri, line, column):
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": column}
        }
        req_id = self.send_request("textDocument/definition", params)
        return self.wait_for_response(req_id)

    def references(self, uri, line, column, include_declaration=True):
        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": column},
            "context": {"includeDeclaration": include_declaration}
        }
        req_id = self.send_request("textDocument/references", params)
        return self.wait_for_response(req_id)

    def shutdown(self):
        req_id = self.send_request("shutdown", {})
        self.wait_for_response(req_id)
        self.send_request("exit", {})
        self.process.terminate()

def uri_to_path(uri):
    parsed = urlparse(uri)
    return unquote(parsed.path)

SYMBOL_KIND_MAP = {
    1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
    6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
    11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
    15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
    20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
    25: "Operator", 26: "TypeParameter"
}
