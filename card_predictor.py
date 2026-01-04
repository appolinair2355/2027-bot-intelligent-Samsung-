# card_predictor.py

import re
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

STATIC_RULES = {
    'A♠️': '❤️', '2♠️': '♣️', '3♠️': '♦️', '4♠️': '♠️',
    'A❤️': '♣️', '2❤️': '♦️', '3❤️': '♠️', '4❤️': '❤️',
    'A♦️': '♠️', '2♦️': '❤️', '3♦️': '♣️', '4♦️': '♦️',
    'A♣️': '♦️', '2♣️': '♠️', '3♣️': '❤️', '4♣️': '♣️'
}

class CardPredictor:
    def __init__(self, telegram_message_sender=None):
        self.telegram_message_sender = telegram_message_sender
        self.predictions = {}
        self.inter_data = []
        self.smart_rules = []
        self.collected_games = set()
        # Configuration automatique forcée
        self.target_channel_id = -1002682552255
        self.prediction_channel_id = -1003554569009
        self.is_inter_mode_active = True # Activé par défaut
        self.auto_prediction_enabled = True
        self.last_predicted_game_number = 0
        self.last_prediction_time = 0.0
        self.prediction_cooldown = 120
        self._last_trigger_used = None
        self._load_all_data()
        # S'assurer que les IDs sont bien ceux demandés même après chargement
        self.target_channel_id = -1002682552255
        self.prediction_channel_id = -1003554569009
        self._save_all_data()

    def _load_all_data(self):
        try:
            if os.path.exists('predictions.json'):
                with open('predictions.json', 'r') as f: self.predictions = json.load(f)
            if os.path.exists('inter_data.json'):
                with open('inter_data.json', 'r') as f: self.inter_data = json.load(f)
            if os.path.exists('smart_rules.json'):
                with open('smart_rules.json', 'r') as f: self.smart_rules = json.load(f)
            if os.path.exists('inter_mode_status.json'):
                with open('inter_mode_status.json', 'r') as f: self.is_inter_mode_active = json.load(f).get('active', True)
            # On ignore config_ids.json pour forcer les IDs codés en dur
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def _save_all_data(self):
        try:
            with open('predictions.json', 'w') as f: json.dump(self.predictions, f)
            with open('inter_data.json', 'w') as f: json.dump(self.inter_data, f)
            with open('smart_rules.json', 'w') as f: json.dump(self.smart_rules, f)
            with open('inter_mode_status.json', 'w') as f: json.dump({'active': self.is_inter_mode_active}, f)
            with open('config_ids.json', 'w') as f:
                json.dump({
                    'target_channel_id': self.target_channel_id,
                    'prediction_channel_id': self.prediction_channel_id
                }, f)
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def extract_game_number(self, text: str) -> Optional[int]:
        # Nettoyage pour ignorer la casse
        text = text.upper()
        # Recherche précise du numéro de jeu avec préfixe N
        m = re.search(r'#N(\d+)', text)
        if m:
            return int(m.group(1))
        # Recherche précise du numéro de jeu entre cercles bleus
        m = re.search(r'🔵(\d+)🔵', text)
        if m:
            return int(m.group(1))
        # Fallback sur n'importe quel nombre précédé de N ou n
        m = re.search(r'N(\d+)', text)
        if m:
            return int(m.group(1))
        return None

    def get_all_cards_in_first_group(self, text: str) -> List[str]:
        # Nettoie le texte pour uniformiser les cœurs et supprimer les espaces parasites
        text = text.replace("❤️", "♥️").replace(" ", "")
        # Recherche toutes les cartes (Valeur + Enseigne) dans tout le message
        # Supporte les emojis avec ou sans variation selector
        return re.findall(r'[AJQK\d]+(?:♠️|♥️|♦️|♣️|♠|❤️|♦|♣)', text)

    def get_first_card_info(self, text: str) -> Optional[Tuple[str, str]]:
        cards = self.get_all_cards_in_first_group(text)
        if not cards: return None
        c = cards[0]
        # Extraction robuste de l'enseigne
        suit = ""
        for s in ['♠️', '♥️', '♦️', '♣️', '♠', '❤️', '♦', '♣']:
            if s in c:
                suit = s
                break
        if not suit: suit = c[-1]
        
        # Uniformisation
        if suit in ["❤️", "❤️️"]: suit = "♥️"
        return c, suit

    def collect_inter_data(self, game_num: int, text: str):
        if game_num in self.collected_games: return
        cards = self.get_all_cards_in_first_group(text)
        if len(cards) < 2: return
        
        # Déclencheur = 1ère carte, Résultat = Enseigne de la 2ème carte
        trigger = cards[0]
        result_card = cards[1]
        result_suit = result_card[-2:] if result_card.endswith(('♠️','♥️','♦️','♣️')) else result_card[-1]
        if result_suit == "❤️": result_suit = "♥️"

        self.inter_data.append({
            'game': game_num, 
            'declencheur': trigger, 
            'result_suit': result_suit,
            'timestamp': time.time()
        })
        self.collected_games.add(game_num)
        self._save_all_data()

    def analyze_and_set_smart_rules(self, chat_id=None, force_activate=False):
        if len(self.inter_data) < 3: return
        
        # On regroupe les résultats par déclencheur pour voir ce qui sort le plus souvent après une carte
        trigger_patterns = defaultdict(Counter)
        for entry in self.inter_data:
            trigger_patterns[entry['declencheur']][entry['result_suit']] += 1
            
        new_rules = []
        for trigger, results in trigger_patterns.items():
            # Pour chaque déclencheur, on prend le résultat le plus fréquent
            suit, count = results.most_common(1)[0]
            # On ne garde que si le déclencheur a été vu au moins 2 fois ou si on a peu de données
            if count >= 1:
                new_rules.append({
                    'trigger': trigger, 
                    'predict': suit, 
                    'count': count,
                    'total': sum(results.values())
                })
        
        # Trier par fiabilité (nombre d'occurrences)
        new_rules.sort(key=lambda x: x['count'], reverse=True)
        
        self.smart_rules = new_rules
        if force_activate: self.is_inter_mode_active = True
        self._save_all_data()

    def should_predict(self, text: str) -> Tuple[bool, Optional[int], Optional[str], bool]:
        if not self.auto_prediction_enabled: return False, None, None, False
        game_num = self.extract_game_number(text)
        if not game_num: return False, None, None, False
        
        # RÈGLE : Écart de exactement 3 par rapport à la dernière prédiction
        # Si on a prédit pour le jeu 900, on attend de voir le jeu 901 pour prédire pour 903 (903-900 = 3)
        # Ou plus simplement: game_num (actuel) + 2 (cible) - last_predicted_game (cible précédente) == 3
        if self.last_predicted_game_number > 0:
            target_game = game_num + 2
            gap = target_game - self.last_predicted_game_number
            if gap != 3:
                # logger.info(f"⏳ Écart non respecté: {gap} (attendu: 3). Cible: {target_game}, Précédent: {self.last_predicted_game_number}")
                return False, None, None, False

        # Extraction du groupe entre parenthèses pour l'analyse des déclencheurs
        first_group_match = re.search(r'(?:\d+)?\(([^)]+)\)', text)
        if first_group_match:
            group_content = first_group_match.group(1)
            cards_to_check = self.get_all_cards_in_first_group(group_content)
        else:
            cards_to_check = self.get_all_cards_in_first_group(text)
            
        if not cards_to_check: return False, None, None, False
        
        # LOGIQUE D'EXCLUSION MUTUELLE
        prediction, is_inter, trigger_used = None, False, None
        
        # 1. SI LE MODE INTER EST ACTIF -> ON N'UTILISE QUE INTER
        if self.is_inter_mode_active:
            if self.smart_rules:
                # Regrouper les règles par enseigne prédite
                rules_by_suit = defaultdict(list)
                for rule in self.smart_rules:
                    rules_by_suit[rule['predict']].append(rule)
                
                # Obtenir les Tops 4 pour chaque enseigne
                top4_by_suit = {}
                for suit, rules in rules_by_suit.items():
                    top4_by_suit[suit] = [r['trigger'] for r in rules[:4]]
                
                # Chercher si une carte du message est dans le Top 4 d'une enseigne
                best_rank = 99
                for card in cards_to_check:
                    card_clean = card.replace("❤️", "♥️")
                    for suit, top4 in top4_by_suit.items():
                        if card_clean in top4:
                            rank = top4.index(card_clean)
                            if rank < best_rank:
                                best_rank = rank
                                prediction = suit
                                trigger_used = card_clean
                                is_inter = True
        
        # 2. SI LE MODE INTER EST INACTIF -> ON N'UTILISE QUE LE STATIQUE (sur la 1ère carte uniquement)
        else:
            info = self.get_first_card_info(text)
            if info:
                card_name = info[0].replace("♥️", "❤️")
                if card_name in STATIC_RULES:
                    prediction, trigger_used, is_inter = STATIC_RULES[card_name], info[0], False
        
        if prediction:
            self._last_trigger_used = trigger_used
            return True, game_num + 2, prediction, is_inter
            
        return False, None, None, False

    def prepare_prediction_text(self, game_num: int, suit: str) -> str:
        return f"🔵{game_num}🔵:{suit}statut :⏳"

    def has_completion_indicators(self, text: str) -> bool:
        return '✅' in text or '❌' in text

    def _verify_prediction_common(self, text: str) -> Dict:
        game_num = self.extract_game_number(text)
        if not game_num: return {}
        
        # Extraction du premier groupe de cartes entre parenthèses
        # Exemple: #N930. 0(J♥️10♥️J♠️) -> le groupe est J♥️10♥️J♠️
        first_group_match = re.search(r'\d+\(([^)]+)\)', text)
        if not first_group_match:
            # Fallback si le format est différent mais qu'on a des cartes
            cards = self.get_all_cards_in_first_group(text)
            first_group_cards = cards[:3] if cards else []
        else:
            group_content = first_group_match.group(1)
            first_group_cards = self.get_all_cards_in_first_group(group_content)

        # On vérifie si ce numéro de jeu correspond à une prédiction en attente (jusqu'à +2)
        target_game = None
        for offset in [0, 1, 2]:
            check_num = game_num - offset
            if str(check_num) in self.predictions:
                pred = self.predictions[str(check_num)]
                if pred.get('status') == 'pending':
                    target_game = str(check_num)
                    break
        
        if not target_game: return {}
        
        pred = self.predictions[target_game]
        predicted_suit = pred['predicted_costume']
        
        # Vérification si le costume prédit est présent dans le premier groupe
        found_in_group = False
        for card in first_group_cards:
            # Extraction de l'enseigne de la carte du groupe
            suit = ""
            for s in ['♠️', '♥️', '♦️', '♣️', '♠', '❤️', '♦', '♣']:
                if s in card:
                    suit = s
                    break
            if not suit: suit = card[-1]
            if suit in ["❤️", "❤️️"]: suit = "♥️"
            
            if suit == predicted_suit:
                found_in_group = True
                break
        
        # Calcul de l'offset pour l'affichage ✅0️⃣, ✅1️⃣, ✅2️⃣
        offset = game_num - int(target_game)
        
        if found_in_group:
            status = 'won'
            symbol = f"✅{chr(0x30 + offset)}️⃣" # Génère ✅0️⃣, ✅1️⃣, ✅2️⃣
        else:
            # Si on est au dernier essai (offset 2) et que c'est toujours pas bon
            if offset >= 2:
                status = 'lost'
                symbol = "❌"
            else:
                # Sinon on attend encore le prochain numéro (n+1 ou n+2)
                return {}

        pred['status'] = status
        self._save_all_data()
        
        return {
            'type': 'edit_message', 
            'message_id_to_edit': pred['message_id'], 
            'new_message': f"🔵{target_game}🔵:{pred['predicted_costume']}statut :{symbol}"
        }

    def get_session_report_preview(self) -> str:
        total = len(self.predictions)
        won = sum(1 for p in self.predictions.values() if p.get('status') == 'won')
        lost = sum(1 for p in self.predictions.values() if p.get('status') == 'lost')
        rate = (won/total*100) if total > 0 else 0
        return f"📊 **BILAN 24h/24**\n\n✅ Gagnés: {won}\n❌ Perdus: {lost}\n📈 Taux: {rate:.1f}%"

    def get_inter_status(self) -> Tuple[str, Dict]:
        is_active = self.is_inter_mode_active
        total_collected = len(self.inter_data)
        
        message = f"🧠 **MODE INTER - {'✅ ACTIF' if is_active else '❌ INACTIF'}**\n\n"
        message += f"📊 {len(self.smart_rules)} règles créées ({total_collected} jeux analysés):\n\n"
        
        # Regrouper par enseigne de prédiction
        rules_by_suit = defaultdict(list)
        for rule in self.smart_rules:
            rules_by_suit[rule['predict']].append(rule)
            
        for suit in ['♠️', '♥️', '♦️', '♣️']:
            if suit in rules_by_suit:
                message += f"**Pour prédire {suit}:**\n"
                # On affiche les 4 meilleures règles par enseigne
                for r in rules_by_suit[suit][:4]:
                    message += f"  • {r['trigger']} ({r['count']}x)\n"
                message += "\n"
        
        kb = {'inline_keyboard': [[{'text': '🔄 Actualiser Analyse', 'callback_data': 'inter_apply'}]]}
        return message, kb
