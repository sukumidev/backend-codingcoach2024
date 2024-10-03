from .resources import Login, CreateUser, ListUsers, RandomQuestion, FilteredRandomQuestion, Logout, AddQuestion, \
    UpdateQuestion, DeleteQuestion, GetAllQuestions, GetUserProfile, DeleteUserProfile, UpdateUserProfile, SubmitAnswer, \
    GetUserResponses, CompileCode, InterviewResource, UserInterviewsResource, UserClassificationResource
from .resources import PreferredLanguagesResource, ScoreProgressResource, ScoreByTechnologyResource, ProfilesPieChartResource, InterviewSummaryResource, InterviewDetailResource


def initialize_routes(api):
    api.add_resource(CreateUser, '/create_user')
    api.add_resource(Login, '/login')
    api.add_resource(ListUsers, '/users')
    api.add_resource(RandomQuestion, '/random-question')
    api.add_resource(FilteredRandomQuestion, '/filtered-random-question')
    api.add_resource(Logout, '/logout')
    api.add_resource(AddQuestion, '/questions')
    api.add_resource(UpdateQuestion, '/questions/<string:id>')
    api.add_resource(DeleteQuestion, '/questions/<string:id>')
    api.add_resource(GetAllQuestions, '/questions/all')
    api.add_resource(GetUserProfile, '/user/profile')
    api.add_resource(UpdateUserProfile, '/user/profile')
    api.add_resource(DeleteUserProfile, '/user/profile')
    api.add_resource(SubmitAnswer, '/submit-answer')
    api.add_resource(GetUserResponses, '/user/responses')
    api.add_resource(CompileCode, '/compile-code')
    api.add_resource(InterviewResource, '/interview')
    api.add_resource(UserInterviewsResource, '/user/interviews')
    api.add_resource(PreferredLanguagesResource, '/user/<string:user_id>/languages')
    api.add_resource(ScoreProgressResource, '/user/<string:user_id>/scores')
    api.add_resource(ScoreByTechnologyResource, '/user/<string:user_id>/technology-scores')
    api.add_resource(ProfilesPieChartResource, '/user/<string:user_id>/profiles-pie')
    api.add_resource(InterviewSummaryResource, '/user/<string:user_id>/interview-summary')
    api.add_resource(InterviewDetailResource, '/user/<string:user_id>/interview/<string:interview_id>')
    api.add_resource(UserClassificationResource, '/clasificar')

