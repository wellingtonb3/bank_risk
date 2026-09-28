import os
import sqlite3
import pandas as pd
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env (onde está sua DEEPSEEK_API_KEY)
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Configura o cliente da DeepSeek (compatível com a API OpenAI)
client = OpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com"
)

def consultar_dados(filtro_faixa):
    conn = sqlite3.connect('bankdataset.db')
    if filtro_faixa == "Todos":
        query = "SELECT * FROM tabela LIMIT 50"
    else:
        query = f"SELECT * FROM tabela WHERE Faixa_Risco = '{filtro_faixa}' LIMIT 50"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def obter_estatisticas():
    conn = sqlite3.connect('bankdataset.db')
    total = pd.read_sql_query("SELECT COUNT(*) as total FROM tabela", conn)['total'][0]
    resumo_risco = pd.read_sql_query(
        "SELECT Faixa_Risco, COUNT(*) as Quantidade FROM tabela GROUP BY Faixa_Risco", conn
    )
    conn.close()
    
    texto_resumo = f"Total de Registros na Base: {total:,}\n\nDistribuição por Faixa de Risco:\n"
    for _, row in resumo_risco.iterrows():
        texto_resumo += f"- {row['Faixa_Risco']}: {row['Quantidade']:,} registros\n"
    return texto_resumo

def perguntar_ao_banco(pergunta_usuario):
    if not DEEPSEEK_API_KEY:
        return "Erro: A chave DEEPSEEK_API_KEY não foi encontrada no arquivo .env."
    
    try:
        # Instrução de sistema para a IA gerar o SQL com base no nosso schema real
        prompt_sistema = """
        Você é um assistente especialista em SQLite. O banco possui uma tabela chamada 'tabela' com as seguintes colunas:
        - Date (TIMESTAMP)
        - Domain (TEXT - ex: INVESTMENTS, EDUCATION)
        - Location (TEXT - Cidade)
        - Value (INTEGER - Valor financeiro)
        - Transaction_count (INTEGER)
        - Dias_Atraso (INTEGER)
        - Score_Credito (INTEGER de 0 a 1000)
        - Faixa_Risco (TEXT)

        O usuário vai fazer uma pergunta sobre esses dados. Você DEVE retornar APENAS uma query SQL válida do SQLite que responda à pergunta, sem markdown extra, sem crases extras, apenas o código SQL puro.
        """
        
        resposta_sql = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pergunta_usuario}
            ],
            stream=False
        )
        
        query_sql = resposta_sql.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
        
        # Executa a query gerada no banco SQLite
        conn = sqlite3.connect('bankdataset.db')
        df_resultado = pd.read_sql_query(query_sql, conn)
        conn.close()
        
        # Pede para a IA traduzir o resultado em uma resposta amigável em português
        prompt_explicacao = f"""
        A pergunta do usuário foi: "{pergunta_usuario}"
        A query SQL executada foi: {query_sql}
        O resultado obtido em formato de dados foi:
        {df_resultado.head(10).to_string()}

        Explique o resultado para o usuário de forma clara, prestativa e em português do Brasil.
        """
        
        resposta_final = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Você é um analista financeiro amigável e prestativo."},
                {"role": "user", "content": prompt_explicacao}
            ],
            stream=False
        )
        
        return f"**Query SQL Gerada pela IA:**\n`{query_sql}`\n\n**Resposta:**\n{resposta_final.choices[0].message.content}"
        
    except Exception as e:
        return f"Ocorreu um erro ao processar a sua pergunta com a IA: {str(e)}"

# Construção da interface visual com Gradio
with gr.Blocks(title="Painel Inteligente de Crédito") as demo:
    gr.Markdown("# 🤖 Painel de Análise de Risco com DeepSeek IA")
    gr.Markdown("Explore sua base de dados de 1 milhão de registros usando inteligência artificial ou filtros tradicionais.")
    
    with gr.Tab("💬 Consultar com IA (DeepSeek)"):
        gr.Markdown("Faça perguntas como: *'Qual a média de score de crédito na cidade de São Paulo?'* ou *'Liste as 5 maiores transações de investimentos.'*")
        input_pergunta = gr.Textbox(label="Sua Pergunta para o Banco de Dados", placeholder="Digite aqui o que deseja saber...")
        btn_perguntar = gr.Button("Perguntar à IA", variant="primary")
        output_resposta = gr.Markdown(label="Resposta do Assistente")
        
        btn_perguntar.click(fn=perguntar_ao_banco, inputs=input_pergunta, outputs=output_resposta)

    with gr.Tab("📊 Visão Geral e Estatísticas"):
        btn_stats = gr.Button("Carregar Resumo da Base")
        output_stats = gr.Textbox(label="Estatísticas Gerais", lines=8)
        btn_stats.click(fn=obter_estatisticas, outputs=output_stats)
        
    with gr.Tab("🔍 Filtro por Faixa de Risco"):
        dropdown_risco = gr.Dropdown(
            choices=[
                "Todos", 
                "Excelente / Risco Muito Baixo", 
                "Bom / Risco Baixo", 
                "Regular / Risco Médio", 
                "Baixo / Risco Alto"
            ],
            label="Selecione a Faixa de Risco",
            value="Todos"
        )
        btn_busca = gr.Button("Buscar Clientes")
        tabela_resultado = gr.Dataframe(label="Amostra de Registros Correspondentes")
        
        btn_busca.click(fn=consultar_dados, inputs=dropdown_risco, outputs=tabela_resultado)

if __name__ == "__main__":
    # Mantém o link público ativo para acesso prático
    demo.launch(share=True)
