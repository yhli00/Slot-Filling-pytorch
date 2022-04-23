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

# slot2desp = {'playlist': 'playlist', 'music_item': 'music item', 'geographic_poi': 'geographic position', 'facility': 'facility', 'movie_name': 'movie name', 
#              'location_name': 'location name', 'restaurant_name': 'restaurant name', 'track': 'track', 'restaurant_type': 'restaurant type', 
#              'object_part_of_series_type': 'series', 'country': 'country', 'service': 'service', 'poi': 'position', 'party_size_description': 'person', 
#              'served_dish': 'served dish', 'genre': 'genre', 'current_location': 'current location', 'object_select': 'this current', 'album': 'album', 
#              'object_name': 'object name', 'state': 'location', 'sort': 'type', 'object_location_type': 'location type', 'movie_type': 'movie type', 
#              'spatial_relation': 'spatial relation', 'artist': 'artist', 'cuisine': 'cuisine', 'entity_name': 'entity name', 'object_type': 'object type', 
#              'playlist_owner': 'owner', 'timeRange': 'time range', 'city': 'city', 'rating_value': 'rating value', 'best_rating': 'best rating', 
#              'rating_unit': 'rating unit', 'year': 'year', 'party_size_number': 'number', 'condition_description': 'weather', 'condition_temperature': 'temperature'}




slot2desp = {
    # AddToPlaylist
    'playlist': 'play list is a list of recorded songs or pieces of music chosen to be broadcast on a radio show or by a particular radio station',
    'music_item': 'music item is musical term',
    'playlist_owner': 'owner is a person who owns something',
    'entity_name': 'entity name is entity name',
    'artist': 'artist is musician a person who plays a musical instrument especially as a profession or is musically talented',
    # PlayMusic
    'genre': 'genre is a category of artistic composition as in music or literature characterized by similarities in form style or subject matter',
    'service': 'service is service or application',
    'year': 'year is year number',
    'album': 'album is a blank book for the insertion of photographs stamps or pictures',
    'sort': 'sort is sort',
    'track': 'track is a recording of one song or piece of music',
    # GetWeather
    'city': 'city is a large town',
    'state': 'state is the civil government of a country',
    'timeRange': 'time range is a point of time as measured in hours and minutes past midnight or noon',
    'current_location': 'current location is current position',
    'country': 'country is a nation with its own government occupying a particular territory',
    'spatial_relation': 'spatial relation is spatial near or far',
    'geographic_poi': 'geographic position is a place where someone or something is located or has been put',
    'condition_temperature': 'temperature is the degree or intensity of heat present in a substance or object especially as expressed according to a comparative scale and shown by a thermometer or perceived by touch',
    'condition_description': 'weather is the state of the atmosphere at a place and time as regards heat dryness sunshine wind rain etc',
    # BookRestaurant
    'facility': 'facility is a place amenity or piece of equipment provided for a particular purpose',
    'restaurant_name': 'restaurant name is restaurant name',
    'cuisine': 'cuisine is a style or method of cooking especially as characteristic of a particular country region or establishment',
    'restaurant_type': 'restaurant type is restaurant type',
    'served_dish': 'served dish is served dish',
    'party_size_number': 'number is number integer digit',
    'poi': 'position is a place where someone or something is located or has been put',
    'party_size_description': 'person is what or which person or people',
    # RateBook
    'object_part_of_series_type': 'series is a set of related television or radio programs especially of a specified kind',
    'object_select': 'this the referential pronoun',
    'rating_value': 'rating value is number integer digit',
    'object_name': 'object name is name',
    'object_type': 'object type is type',
    'rating_unit': 'rating unit is stars',
    'best_rating': 'best rating is highest score',
    # SearchCreativeWork
    # SearchScreeningEvent
    'movie_type': 'movie type is film type',
    'object_location_type': 'location type is cinema theater',
    'location_name': 'location name is cinema name',
    'movie_name': 'movie name is movie name'

}

domain_set = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]

# domain2slots['atis'] = []
# with open("../data/atis/labels.txt", 'r') as fr:
#     for line in fr:
#         slot, desp = line.strip('\n').split('\t')[:2]
#         slot2desp[slot] = desp
#         domain2slots['atis'].append(slot)

