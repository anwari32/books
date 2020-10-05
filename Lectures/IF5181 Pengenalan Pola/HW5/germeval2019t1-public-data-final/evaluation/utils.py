#!/usr/bin/python
# coding: utf8

# Copyright 2019 Language Technology Group, Universität Hamburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Filename: utils.py
# Authors:  Rami Aly, Steffen Remus and Chris Biemann
# Description: Util methods for CodaLab's evaluation script for GermEval-2019 Task 1: Shared task on hierarchical classification of blurbs.
#    For more information visit https://competitions.codalab.org/competitions/21226.
# Requires: sklearn


from __future__ import print_function
import sys
import os.path
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import warnings


#Explicitly lists all genres as training data not necessairly have all labels in test, so we want to avoid an error later
all_labels_set = {0: [u"Ratgeber", u"Kinderbuch & Jugendbuch", u"Literatur & Unterhaltung", u"Sachbuch", u"Ganzheitliches Bewusstsein", u"Architektur & Garten", u"Glaube & Ethik", u"Künste"],
1: [u"Eltern & Familie", u"Echtes Leben, Realistischer Roman", u"Abenteuer", u"Märchen, Sagen", u"Lyrik, Anthologien, Jahrbücher", u"Frauenunterhaltung", u"Fantasy", u"Kommunikation & Beruf", u"Lebenshilfe & Psychologie", u"Krimi & Thriller", u"Freizeit & Hobby", u"Liebe, Beziehung und Freundschaft", u"Familie", u"Natur, Wissenschaft, Technik", u"Fantasy und Science Fiction", u"Geister- und Gruselgeschichten", u"Schicksalsberichte", u"Romane & Erzählungen", u"Science Fiction", u"Politik & Gesellschaft", u"Ganzheitliche Psychologie", u"Natur, Tiere, Umwelt, Mensch", u"Psychologie", u"Lifestyle", u"Sport", u"Lebensgestaltung", u"Essen & Trinken", u"Gesundheit & Ernährung", u"Kunst, Musik", u"Architektur", u"Biographien & Autobiographien", u"Romance", u"Briefe, Essays, Gespräche", u"Kabarett & Satire", u"Krimis und Thriller", u"Erotik", u"Historische Romane", u"Theologie", u"Beschäftigung, Malen, Rätseln", u"Schulgeschichten", u"Biographien", u"Kunst", u"(Zeit-) Geschichte", u"Ganzheitlich Leben", u"Garten & Landschaftsarchitektur", u"Körper & Seele", u"Energieheilung", u"Abenteuer, Reisen, fremde Kulturen", u"Historische Romane, Zeitgeschichte", u"Klassiker & Lyrik", u"Fotografie", u"Design", u"Beauty & Wellness", u"Kunst & Kultur", u"Mystery", u"Ratgeber Partnerschaft & Sexualität", u"Detektivgeschichten", u"Spiritualität & Religion", u"Sachbuch Philosophie", u"Tiergeschichten", u"Horror", u"Literatur & Unterhaltung Satire", u"Infotainment & erzählendes Sachbuch", u"Fitness & Sport", u"Übernatürliches", u"Psychologie & Spiritualität", u"Handwerk Farbe", u"Weisheiten der Welt", u"Naturheilweisen", u"Lustige Geschichten, Witze", u"Wissen & Nachschlagewerke", u"Sterben, Tod und Trauer", u"Romantasy", u"Wirtschaft & Recht", u"Comic & Cartoon", u"Schullektüre", u"Glaube und Grenzerfahrungen", u"Mode & Lifestyle", u"Mondkräfte", u"Musik", u"Geschichte, Politik", u"Gemeindearbeit", u"Wohnen & Innenarchitektur", u"Esoterische Romane", u"Schicksalsdeutung", u"Religionsunterricht", u"Religiöse Literatur", u"Geld & Investment", u"Sportgeschichten", u"Religion, Glaube, Ethik, Philosophie", u"Recht & Steuern", u"Handwerk Holz", u"Regionalia"],
2: [u"Vornamen", u"Heroische Fantasy", u"Joballtag & Karriere", u"Psychothriller", u"Große Gefühle", u"Feiern & Feste", u"Medizin & Forensik", u"Phantastik", u"Ökologie / Umweltschutz", u"Aktuelle Debatten", u"Ganzheitliche Psychologie Lebenshilfe", u"Nordamerikanische Literatur", u"Babys & Kleinkinder", u"Schwangerschaft & Geburt", u"Tod & Trauer", u"Nordische Krimis", u"Gesunde Ernährung", u"Junge Literatur", u"Kreatives", u"Einfamilienhausbau", u"Künstler, Dichter, Denker", u"Themenkochbuch", u"Abenteuer & Action", u"Science Thriller", u"Justizthriller", u"Besser leben", u"Starke Frauen", u"Gesellschaftskritik", u"Psychologie Partnerschaft & Sexualität", u"Krankheit", u"Abenteuer-Fantasy", u"Kirchen- und Theologiegeschichte", u"Biblische Theologie AT", u"Biblische Theologie NT", u"Politik & Gesellschaft Andere Länder & Kulturen", u"Hard Science Fiction", u"All Age Fantasy", u"Trauma", u"Krisen & Ängste", u"Space Opera", u"19./20. Jahrhundert", u"Agenten-/Spionage-Thriller", u"Französische Literatur", u"Selbstcoaching", u"Kopftraining", u"Erzählungen & Kurzgeschichten", u"Gartengestaltung", u"Weltpolitik & Globalisierung", u"Internet", u"Geschenkbuch & Briefkarten", u"Reiseberichte", u"Literatur aus Spanien und Lateinamerika", u"Romantische Komödien", u"Märchen, Legenden und Sagen", u"Humorvolle Unterhaltung", u"Natur, Wissenschaft, Technik Tiere", u"Familiensaga", u"Wellness", u"Romanbiographien", u"Patientenratgeber", u"Politische Theorien", u"Erotik & Sex", u"Rätsel & Spiele", u"Politiker", u"Future-History", u"Gerichtsmedizin / Pathologie", u"Spirituelles Leben", u"Nationalsozialismus", u"Musterbriefe & Rhetorik", u"Einzelthemen der Theologie", u"Dystopie", u"Lyrik", u"Literatur aus Russland und Osteuropa", u"Regionalkrimis", u"Starköche", u"Yoga, Pilates & Stretching", u"Pflanzen & Garten", u"Jenseits & Wiedergeburt", u"Fitnesstraining", u"Problemzonen", u"Italienische Literatur", u"Christlicher Glauben", u"Handwerk Farbe Praxis", u"Handwerk Farbe Grundlagenwissen", u"Östliche Weisheit", u"Ernährung", u"Magen & Darm", u"Nahrungsmittelintoleranz", u"Deutschsprachige Literatur", u"Mittelalter", u"Historische Krimis", u"Kindererziehung", u"Körpertherapien", u"High Fantasy", u"Science Fiction Sachbuch", u"Pubertät", u"Länderküche", u"Styling", u"Schönheitspflege", u"Getränke", u"Lady-Thriller", u"Abschied, Trauer, Neubeginn", u"Laufen & Nordic Walking", u"Neue Wirtschaftsmodelle", u"Utopie", u"Afrikanische Literatur", u"Science Fiction Science Fantasy", u"Englische Literatur", u"Steampunk", u"Alternativwelten", u"Geschichte nach '45", u"Spiritualität & Religion Weltreligionen", u"Theologie Religionspädagogik", u"Raucherentwöhnung", u"Funny Fantasy", u"Skandinavische Literatur", u"Film & Musik", u"Westliche Wege", u"Entspannung & Meditation", u"Kindergarten & Pädagogik", u"Schule & Lernen", u"Spiele & Beschäftigung", u"Psychologie Lebenshilfe", u"Persönlichkeitsentwicklung", u"Mystery-Thriller", u"Homöopathie & Bachblüten", u"Liebe & Beziehung", u"Literaturgeschichte / -kritik", u"Ernährung & Kochen", u"Wandern & Bergsteigen", u"Sucht & Abhängigkeit", u"Politthriller", u"Sterbebegleitung & Hospizarbeit", u"50 plus", u"Job & Karriere", u"Konfirmation", u"Gemeindearbeit Religionspädagogik", u"Kasualien und Sakramente", u"Schauspieler, Regisseure", u"Praktische Anleitungen", u"Rücken & Gelenke", u"Unternehmen & Manager", u"Landschaftsgestaltung", u"Krimikomödien", u"Musiker, Sänger", u"Freizeit & Hobby Tiere", u"Gebete und Andachten", u"Glauben mit Kindern", u"Dark Fantasy", u"Lesen & Kochen", u"Kunst & Kunstgeschichte", u"Flirt & Partnersuche", u"Partnerschaft & Sex", u"Kommunikation", u"Wissen der Naturvölker", u"Urban Fantasy", u"Andere Länder", u"21. Jahrhundert", u"Engel & Schutzgeister", u"Chakren & Aura", u"Science Fiction Satire", u"Bauherrenratgeber", u"Bautechnik", u"Systematische Theologie", u"Praktische Theologie", u"Kosmologie", u"Literatur aus Fernost", u"Bibeln & Katechismus", u"Humoristische Nachschlagewerke", u"Wohnen", u"Länder, Städte & Regionen", u"Spirituelle Entwicklung", u"Indische Literatur", u"Cyberpunk", u"Wissenschaftler", u"Dying Earth", u"Monographien", u"Gesang- und Liederbücher", u"Innenarchitektur", u"Baumaterialien", u"Antike und neulateinische Literatur", u"Gemeindearbeit mit Kindern & Jugendlichen", u"Wissenschaftsthriller", u"Ökothriller", u"Fantasy Science Fantasy", u"Psychotherapie", u"Farbratgeber", u"Hausmittel", u"Schicksalsberichte Andere Länder & Kulturen", u"Design / Lifestyle", u"Diakonie und Seelsorge", u"Gemeindearbeit Sachbuch", u"Gottesdienst und Predigt", u"Sprache & Sprechen", u"(Zeit-) Geschichte Andere Länder & Kulturen", u"Arbeitstechniken", u"Mantras & Mudras", u"NS-Zeit & Nachkriegszeit", u"Kinderschicksal", u"Altbausanierung / Denkmalpflege", u"Neuere Geschichte", u"Umgangsformen", u"Geschichte und Theorie", u"Familie & Religion", u"Niederländische Literatur", u"Handwerk Farbe Gestaltung", u"Historische Fantasy", u"Alte Geschichte", u"Fantasy-/SF-Thriller", u"Bewerbung", u"Wirtschaftsthriller", u"Bibel in gerechter Sprache", u"Fahrzeuge / Technik", u"Handwerk Holz Gestaltung", u"Handwerk Holz Grundlagenwissen", u"Anthologien", u"Handwerk Holz Praxis", u"Bibeln & Bibelarbeit", u"Theologie Weltreligionen", u"Dialog der Traditionen", u"Magie & Hexerei", u"Tierkrimis", u"Medizinthriller", u"Literatur des Nahen Ostens", u"Kirchenthriller", u"Spielewelten", u"Astrologie & Sternzeichen", u"Stadtplanung", u"Feministische Theologie", u"Entwurfs- und Detailplanung", u"Street Art", u"Trennung", u"Philosophie", u"Tarot", u"Systemische Therapie & Familienaufstellung", u"Bauaufgaben", u"Griechische Literatur", u"Gartendesigner", u"Urgeschichte", u"Reden & Glückwünsche", u"Antiquitäten", u"Theater / Ballett"]}


def subtask_A_evaluation(y_true, y_pred):
    all_labels_raw = all_labels_set[0]
    pred_new = []
    for labels in y_pred:
        labels_new = []
        for label in labels:
            if label not in all_labels_raw:
                message = "WARNING: KeyError detected, label %s is unknown to system of subtask A" %str(label)
                print(message, file=sys.stderr)
            else:
                labels_new.append(label)
        pred_new.append(labels_new)

    all_labels = []
    for label in all_labels_raw:
        all_labels.append([label])

    m = MultiLabelBinarizer().fit(all_labels)
    y_true = m.transform(y_true)
    y_pred = m.transform(pred_new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(y_true, y_pred, average = 'micro')
        precision = precision_score(y_true, y_pred, average = 'micro')
        recall = recall_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)

    return[recall, precision, f1, accuracy]


def subtask_B_evaluation(y_true, y_pred):
    all_labels_raw = all_labels_set[0] + all_labels_set[1] + all_labels_set[2]
    pred_new = []
    for labels in y_pred:
        labels_new = []
        for label in labels:
            if label not in all_labels_raw:
                message = "WARNING: KeyError detected, label %s is unknown to system of subtask B" %str(label)
                print(message, file=sys.stderr)
            else:
                labels_new.append(label)
        pred_new.append(labels_new)

    all_labels = []
    for label in all_labels_raw:
        all_labels.append([label])

    m = MultiLabelBinarizer().fit(all_labels)
    y_true = m.transform(y_true)
    y_pred = m.transform(pred_new)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = f1_score(y_true, y_pred, average = 'micro')
        precision = precision_score(y_true, y_pred, average = 'micro')
        recall = recall_score(y_true, y_pred, average = 'micro')
        accuracy = accuracy_score(y_true, y_pred)

    return[recall, precision, f1, accuracy]
