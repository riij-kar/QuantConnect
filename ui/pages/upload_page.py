import os, base64, io
from dash import html, dcc, Input, Output, State, dash_table
from dash import ALL, ctx
import pandas as pd
from utils.fs_helpers import list_dir, ensure_folder, save_csv, DATA_ROOT, is_dir_path, safe_join
from utils.conversion import convert_equity_minute_to_lean_stream, sanitize_to_strict_csv

# Layout builder

def get_upload_layout():
    return html.Div([
        html.H3('Data Upload & Converter'),
        html.Div([
            html.Label('Current Path'),
            dcc.Input(id='fs-current-path', value='', placeholder='relative path', style={'width':'50%'}),
            html.Button('Back', id='fs-back-btn', n_clicks=0, style={'marginLeft':'6px'}),
            html.Button('Refresh', id='fs-refresh-btn', style={'marginLeft':'6px'}),
            html.Button('New Folder', id='fs-new-folder-btn', n_clicks=0, style={'marginLeft':'8px'}),
            dcc.Input(id='fs-new-folder-name', placeholder='folder name', style={'width':'25%', 'marginLeft':'4px'}),
        ], style={'marginBottom':'10px'}),
        html.Div(id='fs-browser'),
        html.Hr(),
        html.H4('Upload CSV'),
        dcc.Upload(id='csv-uploader', children=html.Div(['Drag & Drop or ', html.B('Select CSV')]),
                   multiple=False, style={'border':'1px dashed #999','padding':'20px','width':'50%'}),
        html.Div(id='upload-status', style={'marginTop':'10px'}),
    ])

# Callback registration

def register_upload_callbacks(app):
    # Callback to navigate when a folder link is clicked
    @app.callback(
        Output('fs-current-path','value'),
        Input({'type':'fs-link','path':ALL},'n_clicks'),
        State({'type':'fs-link','path':ALL},'id'),
        State('fs-current-path','value'),
        prevent_initial_call=True
    )
    def navigate_folder(clicks, ids, current):
        if not clicks or not ids or not any(clicks):
            return current or ''
        # Find which link was clicked (has non-None, non-zero n_clicks)
        for click_count, id_dict in zip(clicks, ids):
            if click_count:
                path = id_dict.get('path', '')
                print(f"DEBUG: Link clicked, navigating to {path}")
                return path
        return current or ''
    
    @app.callback(Output('fs-browser','children'), Input('fs-refresh-btn','n_clicks'), Input('fs-current-path','value'))
    def refresh_browser(_, path):
        path = path or ''
        entries = list_dir(path)
        if not entries:
            return html.Div('No contents', style={'color':'#666','fontStyle':'italic'})
        rows = []
        for e in entries:
            if e['is_dir']:
                link = html.Button(e['name'], id={'type':'fs-link','path':e['path']}, n_clicks=0,
                                  style={'color':'#06c', 'background':'none', 'border':'none', 
                                         'cursor':'pointer', 'textDecoration':'underline', 'padding':'0'})
            else:
                link = html.Span(e['name'], style={'color':'#666'})
            rows.append(html.Tr([
                html.Td(link),
                html.Td('DIR' if e['is_dir'] else 'FILE'),
                html.Td(e['path'])
            ]))
        table = html.Table([
            html.Thead(html.Tr([html.Th('Name'), html.Th('Type'), html.Th('Rel Path')])),
            html.Tbody(rows)
        ], style={'width':'70%','fontSize':'12px'})
        return table

    @app.callback(
        Output('fs-current-path','value', allow_duplicate=True),
        Input('fs-back-btn','n_clicks'),
        State('fs-current-path','value'),
        prevent_initial_call=True
    )
    def navigate_back(back_clicks, current):
        if not back_clicks:
            return current or ''
        cur = (current or '').strip('/')
        if not cur:
            return ''
        parts = cur.split('/')
        parent = '/'.join(parts[:-1])
        print(f"DEBUG: Back button clicked, navigating from {current} to {parent}")
        return parent
    
    @app.callback(
        Output('fs-current-path','value', allow_duplicate=True),
        Input('fs-new-folder-btn','n_clicks'),
        State('fs-new-folder-name','value'),
        State('fs-current-path','value'),
        prevent_initial_call=True
    )
    def create_new_folder(new_clicks, folder_name, current):
        if not new_clicks or not folder_name:
            return current or ''
        try:
            new_path = os.path.join(current or '', folder_name).replace('\\','/')
            ensure_folder(new_path)
            print(f"DEBUG: Created folder {new_path}")
            return new_path
        except Exception as e:
            print(f"Error creating folder: {e}")
            return current or ''

    @app.callback(Output('upload-status','children'), Input('csv-uploader','contents'), State('csv-uploader','filename'), State('fs-current-path','value'))
    def save_upload(content, filename, path):
        if not content or not filename:
            return ''
        if not filename.lower().endswith('.csv'):
            return html.Span('Only CSV files supported.', style={'color':'red'})
        rel_path = (path or '').strip('/')
        symbol = rel_path.split('/')[-1] if rel_path else None
        if not symbol:
            return html.Span('Select a symbol folder first (e.g. equity/india/minute/SYMBOL).', style={'color':'red'})
        try:
            header, b64 = content.split(',', 1)
            raw_bytes = base64.b64decode(b64)
            abs_base = safe_join(DATA_ROOT, rel_path)
            os.makedirs(abs_base, exist_ok=True)
            sanitized_csv = sanitize_to_strict_csv(raw_bytes)
            original_path = os.path.join(abs_base, f'original_{filename}')
            with open(original_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(sanitized_csv)
            result = convert_equity_minute_to_lean_stream(original_path, symbol)
            mode = result.get('mode')
            files = result.get('files', [])
            if mode == 'daily':
                if not files:
                    return html.Span('Daily upload detected but no rows converted.', style={'color':'orange'})
                created_zip = files[0]
                daily_zip_path = os.path.join(abs_base, created_zip)
                symbol_safe = (symbol or '').strip().replace(' ', '_').lower()
                base_tail = os.path.basename(abs_base).strip().replace(' ', '_').lower()
                if os.path.isfile(daily_zip_path) and symbol_safe and base_tail == symbol_safe:
                    dest_dir = os.path.dirname(abs_base)
                    dest_path = os.path.join(dest_dir, created_zip)
                    try:
                        os.replace(daily_zip_path, dest_path)
                        created_zip = os.path.relpath(dest_path, DATA_ROOT).replace('\\', '/')
                    except OSError:
                        pass
                return html.Span(
                    f'Daily data converted -> {created_zip} (includes lean CSV and original_{filename}).',
                    style={'color':'green'}
                )
            if not files:
                return html.Span('No valid minute rows detected. Nothing converted.', style={'color':'orange'})
            preview = ', '.join(files[:5]) + (' ...' if len(files) > 5 else '')
            return html.Span(f'Converted {len(files)} day(s): {preview}', style={'color':'green'})
        except Exception as e:
            return html.Span(f'Upload failed: {e}', style={'color':'red'})

    # new-folder handled in combined callback above

