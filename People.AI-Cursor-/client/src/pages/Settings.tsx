import * as React from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Eye, EyeOff, User, Lock, Bell, Globe, Loader2, Edit, Save, X } from "lucide-react"
import { criteriaApi, type FitmentCriteria } from "@/lib/api"
import { toast } from "@/components/ui/use-toast"

export default function Settings() {
  const [showPassword, setShowPassword] = React.useState(false)
  const [settings, setSettings] = React.useState({
    emailNotifications: true,
    pushNotifications: false,
    darkMode: true,
    language: 'en',
  })

  const [fitmentCriteria, setFitmentCriteria] = React.useState<FitmentCriteria | null>(null)
  const [isEditing, setIsEditing] = React.useState(false)
  const [isLoading, setIsLoading] = React.useState(true)
  const [editValues, setEditValues] = React.useState({
    best_fit: 0,
    average_fit: 0,
    not_fit: 0,
  })

  const handleSettingChange = (name: string, value: boolean | string) => {
    setSettings(prev => ({
      ...prev,
      [name]: value
    }))
  }

  // Fetch criteria on component mount
  React.useEffect(() => {
    const fetchCriteria = async () => {
      try {
        setIsLoading(true)
        const data = await criteriaApi.getCriteria()
        setFitmentCriteria(data)
        setEditValues({
          best_fit: data.best_fit,
          average_fit: data.average_fit,
          not_fit: data.not_fit,
        })
      } catch (error) {
        console.error('Failed to fetch criteria:', error)
        toast({
          title: "Error",
          description: "Failed to load criteria. Please try again.",
          variant: "destructive",
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchCriteria()
  }, [])

  const handleEditClick = () => {
    setIsEditing(true)
  }

  const handleCancelEdit = () => {
    if (fitmentCriteria) {
      setEditValues({
        best_fit: fitmentCriteria.best_fit,
        average_fit: fitmentCriteria.average_fit,
        not_fit: fitmentCriteria.not_fit,
      })
    }
    setIsEditing(false)
  }

  const handleInputChange = (field: keyof typeof editValues, value: number) => {
    setEditValues(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const handleSaveCriteria = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate the input
    if (editValues.best_fit <= editValues.average_fit || 
        editValues.average_fit <= editValues.not_fit || 
        editValues.not_fit < 0) {
      toast({
        title: "Invalid values",
        description: "Please ensure: Best Fit > Average Fit > Not Fit ≥ 0",
        variant: "destructive",
      })
      return
    }

    try {
      const updatedCriteria = await criteriaApi.updateCriteria(editValues)
      setFitmentCriteria(updatedCriteria)
      setIsEditing(false)
      toast({
        title: "Success",
        description: "Criteria updated successfully",
      })
    } catch (error) {
      console.error('Failed to update criteria:', error)
      toast({
        title: "Error",
        description: "Failed to update criteria. Please try again.",
        variant: "destructive",
      })
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Settings updated:', settings)
  }

  return (
    <div className="container mx-auto py-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">Settings</h1>
      </div>

      <Tabs defaultValue="profile" className="space-y-8">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="profile" className="flex items-center gap-2">
            <User className="h-4 w-4" />
            <span>Profile</span>
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Lock className="h-4 w-4" />
            <span>Security</span>
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            <span>Notifications</span>
          </TabsTrigger>
          <TabsTrigger value="criteria" className="flex items-center gap-2">
            <Globe className="h-4 w-4" />
            <span>Criteria</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <CardTitle>Profile Settings</CardTitle>
              <CardDescription>Update your profile information</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <form onSubmit={handleSubmit}>
                <div className="space-y-6">
                  <div className="flex flex-col items-center gap-4">
                    <Avatar className="h-24 w-24">
                      <AvatarImage src="https://github.com/shadcn.png" alt="Profile" />
                      <AvatarFallback>CN</AvatarFallback>
                    </Avatar>
                    <Button variant="outline" type="button">
                      Change Photo
                    </Button>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Label htmlFor="firstName">First Name</Label>
                      <Input id="firstName" defaultValue="John" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="lastName">Last Name</Label>
                      <Input id="lastName" defaultValue="Doe" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="email">Email</Label>
                    <Input id="email" type="email" defaultValue="john.doe@example.com" />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="bio">Bio</Label>
                    <Textarea id="bio" placeholder="Tell us about yourself..." className="min-h-[100px]" />
                  </div>

                  <Button type="submit">Save Changes</Button>
                </div>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security">
          <Card>
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>Update your password and security settings</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="currentPassword">Current Password</Label>
                  <div className="relative">
                    <Input 
                      id="currentPassword" 
                      type={showPassword ? "text" : "password"} 
                      className="pr-10"
                    />
                    <Button 
                      type="button" 
                      variant="ghost" 
                      size="icon" 
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? 
                        <EyeOff className="h-4 w-4" /> : 
                        <Eye className="h-4 w-4" />
                      }
                      <span className="sr-only">Toggle password visibility</span>
                    </Button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="newPassword">New Password</Label>
                    <Input id="newPassword" type={showPassword ? "text" : "password"} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword">Confirm New Password</Label>
                    <Input id="confirmPassword" type={showPassword ? "text" : "password"} />
                  </div>
                </div>

                <Button type="submit">Update Password</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle>Notification Preferences</CardTitle>
              <CardDescription>Manage your notification settings</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="flex items-center justify-between space-x-2">
                  <div className="space-y-0.5">
                    <Label htmlFor="email-notifications">Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive email notifications for important updates
                    </p>
                  </div>
                  <Switch 
                    id="email-notifications" 
                    checked={settings.emailNotifications}
                    onCheckedChange={(checked) => handleSettingChange('emailNotifications', checked)}
                  />
                </div>

                <div className="flex items-center justify-between space-x-2">
                  <div className="space-y-0.5">
                    <Label htmlFor="push-notifications">Push Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable push notifications in your browser
                    </p>
                  </div>
                  <Switch 
                    id="push-notifications" 
                    checked={settings.pushNotifications}
                    onCheckedChange={(checked) => handleSettingChange('pushNotifications', checked)}
                  />
                </div>

                <Button type="submit">Save Preferences</Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="criteria">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Fitment Criteria</CardTitle>
                  <CardDescription>Set the thresholds for candidate fitment scoring</CardDescription>
                </div>
                {!isEditing ? (
                  <Button 
                    type="button" 
                    variant="outline" 
                    size="sm"
                    onClick={handleEditClick}
                    disabled={!fitmentCriteria || isLoading}
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit Criteria
                  </Button>
                ) : (
                  <div className="space-x-2">
                    <Button 
                      type="button" 
                      variant="outline" 
                      size="sm"
                      onClick={handleCancelEdit}
                    >
                      <X className="h-4 w-4 mr-2" />
                      Cancel
                    </Button>
                    <Button 
                      type="submit" 
                      form="criteria-form"
                      size="sm"
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Save className="h-4 w-4 mr-2" />
                      )}
                      Save Changes
                    </Button>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex justify-center items-center h-40">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : !fitmentCriteria ? (
                <div className="text-center py-8 text-muted-foreground">
                  Unable to load criteria. Please try again.
                </div>
              ) : (
                <form id="criteria-form" onSubmit={handleSaveCriteria} className="space-y-6">
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="space-y-2">
                        <Label htmlFor="best-fit">Best Fit (≥)</Label>
                        <Input
                          id="best-fit"
                          type="number"
                          value={editValues.best_fit}
                          onChange={(e) => handleInputChange('best_fit', Number(e.target.value))}
                          disabled={!isEditing}
                          min={editValues.average_fit + 1}
                        />
                        <p className="text-xs text-muted-foreground">
                          Score ≥ {editValues.best_fit}%
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="average-fit">Average Fit (≥)</Label>
                        <Input
                          id="average-fit"
                          type="number"
                          value={editValues.average_fit}
                          onChange={(e) => handleInputChange('average_fit', Number(e.target.value))}
                          disabled={!isEditing}
                          min={editValues.not_fit + 1}
                          max={editValues.best_fit - 1}
                        />
                        <p className="text-xs text-muted-foreground">
                          {editValues.best_fit}% &gt; Score ≥ {editValues.average_fit}%
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="not-fit">Not Fit (≤)</Label>
                        <Input
                          id="not-fit"
                          type="number"
                          value={editValues.not_fit}
                          onChange={(e) => handleInputChange('not_fit', Number(e.target.value))}
                          disabled={!isEditing}
                          min={0}
                          max={editValues.average_fit - 1}
                        />
                        <p className="text-xs text-muted-foreground">
                          Score &lt; {editValues.average_fit}%
                        </p>
                      </div>
                    </div>

                    {isEditing && (
                      <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-md border border-amber-200 dark:border-amber-800">
                        <p className="text-sm text-amber-800 dark:text-amber-200">
                          Ensure: <strong>Best Fit &gt; Average Fit &gt; Not Fit ≥ 0</strong>
                        </p>
                      </div>
                    )}
                  </div>
                </form>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}