domain2slots = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 
                       'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album', 'sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type', 'object_type', 'location_name', 'spatial_relation', 'movie_name']
}

slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'movie name', 
             'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 
             'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 
             'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 
             'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 
             'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 
             'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 
             'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

domain2slots['atis'] = []
with open("data/atis/labels.txt", 'r') as fr:
    for line in fr:
        slot, desp = line.strip('\n').split('\t')[:2]
        slot2desp[slot] = desp
        domain2slots['atis'].append(slot)

