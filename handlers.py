# handlers.py

import logging
import time
import json
from collections import defaultdict
from typing import Dict, Any, Optional
import requests
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Importation Robuste
try:
    from card_predictor import CardPredictor
except ImportError:
    logger.error("❌ IMPOSSIBLE D'IMPORTER CARDPREDICTOR")
    CardPredictor = None

user_message_counts = defaultdict(list)

WELCOME_MESSAGE = """
👋 **BIENVENUE SUR LE BOT ENSEIGNE !** ♠️♥️♦️♣️

Je prédis la prochaine Enseigne (Couleur) en utilisant :
1. **Règles statiques** : Patterns prédéfinis
2. **Intelligence artificielle (Mode INTER)** : Apprend des données réelles

━━━━━━━━━━━━━━━━━━━━━
📋 **COMMANDES DISPONIBLES**
━━━━━━━━━━━━━━━━━━━━━

**🔹 Informations Générales**
• `/start` - Afficher ce message d'aide
• `/stat` - Voir l'état du bot (canaux, mode actif)
• `/bilan` - Voir l'aperçu du prochain bilan

**🔹 Mode Intelligent (INTER)**
• `/inter status` - Voir les règles apprises (Top 4 par enseigne)
• `/inter activate` - Activer manuellement l'IA
• `/inter default` - Revenir aux règles statiques

**🔹 Prédictions Automatiques**
• `/auto` - Activer ou désactiver l'envoi automatique

**🔹 Collecte de Données**
• `/collect` - Voir les données collectées par enseigne
• `/reset` - Réinitialiser COMPLÈTEMENT le bot

**🔹 Configuration**
• `/config` - Configurer les canaux (Source/Prédiction)

**🔹 Maintenance**
• `/deploy` - Télécharger le package de déploiement
• `/qua` - État et informatique secret du bot

━━━━━━━━━━━━━━━━━━━━━
💡 **Fonctionnement :** Le bot surveille le canal SOURCE, prédit selon les règles (si un TOP est trouvé) et envoie dans le canal PRÉDICTION.
━━━━━━━━━━━━━━━━━━━━━
"""

HELP_MESSAGE = """
🤖 **AIDE COMMANDE /INTER**

• `/inter status` : Voir les règles apprises (Top 2 par Enseigne).
• `/inter activate` : Forcer l'activation de l'IA et relancer l'analyse.
• `/inter default` : Revenir aux règles statiques.
"""

class TelegramHandlers:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        if CardPredictor:
            self.card_predictor = CardPredictor(telegram_message_sender=self.send_message)
        else:
            self.card_predictor = None

    def _check_rate_limit(self, user_id):
        now = time.time()
        user_message_counts[user_id] = [t for t in user_message_counts[user_id] if now - t < 60]
        user_message_counts[user_id].append(now)
        return len(user_message_counts[user_id]) <= 30

    def send_message(self, chat_id: int, text: str, parse_mode='Markdown', message_id: Optional[int] = None, edit=False, reply_markup: Optional[Dict] = None) -> Optional[int]:
        if not chat_id or not text: return None
        
        method = 'editMessageText' if (message_id or edit) else 'sendMessage'
        payload = {'chat_id': chat_id, 'text': text, 'parse_mode': parse_mode}
        
        if message_id: payload['message_id'] = message_id
        if reply_markup: 
            payload['reply_markup'] = json.dumps(reply_markup) if isinstance(reply_markup, dict) else reply_markup

        try:
            r = requests.post(f"{self.base_url}/{method}", json=payload, timeout=10)
            if r.status_code == 200:
                return r.json().get('result', {}).get('message_id')
            else:
                logger.error(f"Erreur Telegram {r.status_code}: {r.text}")
        except Exception as e:
            logger.error(f"Exception envoi message: {e}")
        return None

    def _handle_command_deploy(self, chat_id: int):
        try:
            zip_filename = 'yoi.zip'
            import os
            self.send_message(chat_id, "📦 **Préparation du package de déploiement...**")
            # Nettoyage des anciens fichiers zip avant création
            os.system("rm -f *.zip")
            # Création du nouveau zip en excluant tout ce qui n'est pas nécessaire
            os.system(f"zip -r {zip_filename} . -x '*.git*' '*__pycache__*' '*.log' 'secrets_config.json' '.replit' 'replit.nix' '*.json'")
            
            if not os.path.exists(zip_filename):
                self.send_message(chat_id, f"❌ Erreur lors de la création de {zip_filename}")
                return
                
            url = f"{self.base_url}/sendDocument"
            with open(zip_filename, 'rb') as f:
                files = {'document': (zip_filename, f, 'application/zip')}
                data = {
                    'chat_id': chat_id,
                    'caption': f'📦 **{zip_filename} - Bot ENSEIGNE v7.1**\n\n✅ Compatible Render.com (Port 10000)\n✅ Nettoyage complet (Sans sessions/quarantaine)\n✅ Mode INTER Strict Top 4\n✅ Écart de 3 jeux respecté',
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, data=data, files=files, timeout=60)
            
            if response.json().get('ok'):
                self.send_message(chat_id, f"✅ **{zip_filename} envoyé avec succès!**")
            else:
                self.send_message(chat_id, f"❌ Erreur Telegram : {response.text}")
        except Exception as e:
            logger.error(f"Erreur /deploy : {e}")
            self.send_message(chat_id, f"❌ Erreur : {str(e)}")

    def _handle_command_collect(self, chat_id: int):
        if not self.card_predictor: 
            self.send_message(chat_id, "❌ Le moteur de prédiction n'est pas chargé.")
            return
        is_active = self.card_predictor.is_inter_mode_active
        total_collected = len(self.card_predictor.inter_data)
        message = "🧠 **ETAT DU MODE INTELLIGENT**\n\n"
        message += f"Actif : {'✅ OUI' if is_active else '❌ NON'}\n"
        message += f"Données collectées : {total_collected}\n\n"
        if self.card_predictor.inter_data:
            from collections import defaultdict
            by_result_suit = defaultdict(list)
            for entry in self.card_predictor.inter_data:
                result_suit = entry.get('result_suit', '?')
                trigger = entry.get('declencheur', '?').replace("♥️", "❤️")
                by_result_suit[result_suit].append(trigger)
            message += "📊 **TOUS LES DÉCLENCHEURS COLLECTÉS:**\n\n"
            for suit in ['♠️', '❤️', '♦️', '♣️']:
                if suit in by_result_suit:
                    triggers = by_result_suit[suit]
                    message += f"**Pour enseigne {suit}:**\n"
                    from collections import Counter
                    trigger_counts = Counter(triggers)
                    for trigger, count in trigger_counts.most_common():
                        message += f"  • {trigger} ({count}x)\n"
                    message += "\n"
        else:
            message += "⚠️ **Aucune donnée collectée.**\n"
        if total_collected < 3:
            message += f"\n⚠️ Minimum 3 jeux requis pour créer des règles (actuellement: {total_collected})."
        keyboard = {'inline_keyboard': []}
        if total_collected >= 3:
            if is_active:
                keyboard['inline_keyboard'].append([
                    {'text': '🔄 Relancer Analyse', 'callback_data': 'inter_apply'},
                    {'text': '❌ Désactiver INTER', 'callback_data': 'inter_default'}
                ])
            else:
                keyboard['inline_keyboard'].append([
                    {'text': '✅ Activer INTER', 'callback_data': 'inter_apply'}
                ])
        else:
            keyboard['inline_keyboard'].append([
                {'text': '🔄 Analyser les données', 'callback_data': 'inter_apply'}
            ])
        self.send_message(chat_id, message, reply_markup=keyboard)

    def _handle_command_bilan(self, chat_id: int):
        if not self.card_predictor:
            self.send_message(chat_id, "❌ Le moteur de prédiction n'est pas chargé.")
            return
        try:
            msg = self.card_predictor.get_session_report_preview()
            keyboard = {'inline_keyboard': [
                [{'text': '✅ Envoyer au canal', 'callback_data': 'send_bilan_confirm'},
                 {'text': '❌ Annuler', 'callback_data': 'config_cancel'}]
            ]}
            self.send_message(chat_id, msg, reply_markup=keyboard)
        except Exception as e:
            logger.error(f"❌ Erreur bilan: {e}")
            self.send_message(chat_id, "❌ Erreur lors de la préparation du bilan.")

    def _handle_command_reset(self, chat_id: int):
        if not self.card_predictor:
            self.send_message(chat_id, "❌ Le moteur de prédiction n'est pas chargé.")
            return
        try:
            cp = self.card_predictor
            cp.predictions = {}
            cp.inter_data = []
            cp.smart_rules = []
            cp.collected_games = set()
            cp._save_all_data()
            self.send_message(chat_id, "✅ RÉINITIALISATION COMPLÈTE EFFECTUÉE")
        except Exception as e:
            logger.error(f"Erreur /reset : {e}")
            self.send_message(chat_id, f"❌ Erreur: {e}")

    def _handle_command_qua(self, chat_id: int):
        if not self.card_predictor:
            self.send_message(chat_id, "❌ Le moteur de prédiction n'est pas chargé.")
            return
        try:
            cp = self.card_predictor
            message = "🔒 **ÉTAT ET INFORMATIQUE SECRET DU BOT**\n\n"
            
            # Afficher les dernières prédictions avec leurs déclencheurs
            if cp.predictions:
                message += "📊 **Les 5 dernières prédictions envoyées**\n"
                sorted_preds = sorted(cp.predictions.items(), key=lambda x: x[1].get('timestamp', 0), reverse=True)
                for game_num, data in sorted_preds[:5]:
                    trigger = data.get('predicted_from_trigger', '?')
                    suit = data.get('predicted_costume', '?')
                    status = data.get('status', 'pending')
                    mode_label = "🧠 INTER" if data.get('is_inter') else "📜 STATIQUE"
                    status_symbol = "✅" if status == 'won' else "❌" if status == 'lost' else "⏳"
                    message += f"  • Jeu {game_num}: {suit} ({status_symbol}) - Déclencheur: {trigger} {mode_label}\n"
            else:
                message += "ℹ️ Aucune prédiction récente.\n"
            
            message += f"\n\n🧠 Mode INTER: {'✅ ACTIF' if cp.is_inter_mode_active else '❌ INACTIF'}\n"
            message += f"\n📈 Donnees collectees: {len(cp.inter_data)} jeux\n"
            message += "📋 Regles UTILISER INTELLIGENT :\n\n"
            
            # Afficher les règles intelligentes regroupées par enseigne
            rules_by_suit = defaultdict(list)
            for rule in cp.smart_rules:
                rules_by_suit[rule['predict']].append(rule)
                
            for suit in ['♠️', '♦️', '♣️', '❤️']: # Ordre demandé
                suit_key = suit.replace('❤️', '♥️')
                rules = rules_by_suit.get(suit, []) or rules_by_suit.get(suit_key, [])
                if rules:
                    message += f"**Pour predire {suit}:**\n"
                    for r in rules[:4]: # Changé à 4
                        message += f"  • {r['trigger']} ({r['count']}x)\n"
                    message += "\n"
                
            self.send_message(chat_id, message)
        except Exception as e:
            logger.error(f"Erreur /qua : {e}")
            self.send_message(chat_id, f"❌ Erreur: {str(e)}")

    def _handle_command_inter(self, chat_id: int, text: str):
        if not self.card_predictor: 
            self.send_message(chat_id, "❌ Le moteur de prédiction n'est pas chargé.")
            return
        parts = text.lower().split()
        action = parts[1] if len(parts) > 1 else 'status'
        if action == 'activate':
            self.card_predictor.analyze_and_set_smart_rules(chat_id=chat_id, force_activate=True)
            self.send_message(chat_id, "✅ **MODE INTER ACTIVÉ**")
        elif action == 'default':
            self.card_predictor.is_inter_mode_active = False
            self.card_predictor._save_all_data()
            self.send_message(chat_id, "❌ **MODE INTER DÉSACTIVÉ**")
        elif action == 'status':
            try:
                msg, kb = self.card_predictor.get_inter_status()
                self.send_message(chat_id, msg, reply_markup=kb)
            except Exception as e:
                logger.error(f"Error in /inter status: {e}")
                self.send_message(chat_id, f"❌ Erreur lors de la récupération du statut: {str(e)}")
        else:
            self.send_message(chat_id, HELP_MESSAGE)

    def handle_update(self, update: Dict[str, Any]):
        try:
            if not self.card_predictor: return
            
            # Log complet de l'update pour le debug
            # logger.info(f"Update received: {json.dumps(update)}")
            
            # Extraction du message de manière plus large
            msg = update.get('message') or update.get('channel_post') or update.get('edited_message') or update.get('edited_channel_post')
            
            # Gérer les updates sans contenu de message direct (comme callback_query)
            if not msg:
                if 'callback_query' in update: 
                    self._handle_callback_query(update['callback_query'])
                return
            
            chat_id = msg.get('chat', {}).get('id')
            text = msg.get('text') or msg.get('caption', '')
            user_id = msg.get('from', {}).get('id', 0)
            
            if not chat_id: return

            # Conversion IDs en string pour comparaison
            source_id = str(self.card_predictor.target_channel_id)
            current_chat_id = str(chat_id)

            logger.info(f"Message de {current_chat_id} (Source attendue: {source_id}) | Texte: {text[:50]}...")
            
            # Gestion des commandes (même sans texte pour certains types d'updates)
            if text and text.startswith('/'):
                logger.info(f"Commande détectée: {text} depuis {chat_id}")
                if text.startswith('/start'): self.send_message(chat_id, WELCOME_MESSAGE)
                elif text.startswith('/inter'): self._handle_command_inter(chat_id, text)
                elif text.startswith('/config'):
                    kb = {'inline_keyboard': [[{'text': 'Source', 'callback_data': 'config_source'}, {'text': 'Prediction', 'callback_data': 'config_prediction'}, {'text': 'Annuler', 'callback_data': 'config_cancel'}]]}
                    self.send_message(chat_id, "⚙️ **CONFIGURATION**", reply_markup=kb)
                elif text.startswith('/stat'):
                    sid = self.card_predictor.target_channel_id or "Non défini"
                    pid = self.card_predictor.prediction_channel_id or "Non défini"
                    mode = "IA" if self.card_predictor.is_inter_mode_active else "Statique"
                    self.send_message(chat_id, f"📊 **STATUS**\nSource: `{sid}`\nPrédiction: `{pid}`\nMode: {mode}")
                elif text.startswith('/deploy'): self._handle_command_deploy(chat_id)
                elif text.startswith('/collect'): self._handle_command_collect(chat_id)
                elif text.startswith('/qua'): self._handle_command_qua(chat_id)
                elif text.startswith('/reset'): self._handle_command_reset(chat_id)
                elif text.startswith('/bilan'): self._handle_command_bilan(chat_id)
                elif text.startswith('/auto'):
                    current = self.card_predictor.auto_prediction_enabled
                    keyboard = {'inline_keyboard': [[{'text': 'Désactiver' if current else 'Activer', 'callback_data': 'toggle_auto_pred'}]]}
                    self.send_message(chat_id, f"🤖 **Auto: {'ON' if current else 'OFF'}**", reply_markup=keyboard)
                return

            if not text: return

            is_edit = 'edited_message' in update or 'edited_channel_post' in update
            
            # Traitement Canal Source
            source_id = str(self.card_predictor.target_channel_id)
            pred_id = str(self.card_predictor.prediction_channel_id)
            current_chat_id = str(chat_id)

            if current_chat_id == source_id:
                logger.info(f"📥 Message du canal SOURCE ({chat_id})")
                game_num = self.card_predictor.extract_game_number(text)
                if game_num:
                    self.card_predictor.collect_inter_data(game_num, text)
                    logger.info(f"📊 Données collectées pour le jeu {game_num}")
                
                # Vérification des prédictions (si indicateur de fin ou emoji spécifique)
                if self.card_predictor.has_completion_indicators(text) or '🔰' in text:
                    res = self.card_predictor._verify_prediction_common(text)
                    if res and res.get('type') == 'edit_message':
                        logger.info(f"✅ Vérification de prédiction pour le jeu {game_num}: {res['new_message']}")
                        self.send_message(self.card_predictor.prediction_channel_id, res['new_message'], message_id=res['message_id_to_edit'], edit=True)
                
                # Nouvelle prédiction si c'est pas un edit
                if not is_edit:
                    ok, num, val, is_inter = self.card_predictor.should_predict(text)
                    if ok and num and val:
                        txt = self.card_predictor.prepare_prediction_text(num, val)
                        mid = self.send_message(self.card_predictor.prediction_channel_id, txt)
                        if mid:
                            trigger = self.card_predictor._last_trigger_used or '?'
                            self.card_predictor.predictions[str(num)] = {
                                'predicted_costume': val, 'predicted_from_trigger': trigger,
                                'message_id': mid, 'timestamp': time.time(), 'status': 'pending', 'is_inter': is_inter
                            }
                            self.card_predictor.last_predicted_game_number = num
                            self.card_predictor.last_prediction_time = time.time()
                            self.card_predictor._save_all_data()
                            logger.info(f"🚀 Prédiction envoyée: Jeu {num}, Suit {val}")
                        else:
                            logger.error(f"❌ Échec de l'envoi de la prédiction au canal {pred_id}")

        except Exception as e:
            logger.error(f"Update error: {e}")

    def _handle_callback_query(self, query: Dict[str, Any]):
        try:
            chat_id = query['message']['chat']['id']
            mid = query['message']['message_id']
            data = query['data']
            callback_id = query['id']
            
            # Répondre au callback pour enlever le sablier sur Telegram
            requests.post(f"{self.base_url}/answerCallbackQuery", json={'callback_query_id': callback_id})
            
            if data == 'toggle_auto_pred' and self.card_predictor:
                self.card_predictor.auto_prediction_enabled = not self.card_predictor.auto_prediction_enabled
                self.card_predictor._save_all_data()
                self.send_message(chat_id, f"✅ Prédictions auto: {'Activées' if self.card_predictor.auto_prediction_enabled else 'Désactivées'}")
            elif data == 'inter_apply' and self.card_predictor:
                self.card_predictor.analyze_and_set_smart_rules(chat_id=chat_id, force_activate=True)
                self.send_message(chat_id, "✅ Analyse terminée et Mode INTER activé !")
            elif data == 'inter_default' and self.card_predictor:
                self.card_predictor.is_inter_mode_active = False
                self.card_predictor._save_all_data()
                self.send_message(chat_id, "✅ Mode INTER désactivé")
            elif data == 'config_source' and self.card_predictor:
                self.card_predictor.target_channel_id = chat_id
                self.card_predictor._save_all_data()
                self.send_message(chat_id, f"✅ Canal SOURCE configuré: `{chat_id}`")
            elif data == 'config_prediction' and self.card_predictor:
                self.card_predictor.prediction_channel_id = chat_id
                self.card_predictor._save_all_data()
                self.send_message(chat_id, f"✅ Canal PRÉDICTION configuré: `{chat_id}`")
            elif data == 'send_bilan_confirm' and self.card_predictor:
                report = self.card_predictor.get_session_report_preview()
                pred_channel = self.card_predictor.prediction_channel_id
                if pred_channel:
                    self.send_message(pred_channel, report)
                    self.send_message(chat_id, "✅ Bilan envoyé au canal de prédiction.")
                else:
                    self.send_message(chat_id, "❌ Canal de prédiction non configuré.")
            elif data == 'config_cancel':
                self.send_message(chat_id, "❌ Action annulée.")
        except Exception as e:
            logger.error(f"Error in callback query: {e}")
